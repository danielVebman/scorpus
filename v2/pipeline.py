from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Step(ABC):
  @abstractmethod
  def extract(self):
    '''
    Extracts any data, stored in `self` for the transform stage.
    '''
    pass

  @abstractmethod
  def transform(self):
    '''
    Performs any transformations on the extracted data.
    '''
    pass

  @abstractmethod
  def load(self):
    '''
    Persists any transformed data.
    '''
    pass

  def run(self):
    '''
    Runs the extract-transform-load process.
    '''
    self.extract()
    self.transform()
    self.load()


class ReadCorpus(Step):
  UTTERANCES_PATH = 'flatfiles/all_utterances.csv'
  CONVERSATIONS_PATH = 'flatfiles/all_conversations.csv'
  SPEAKERS_PATH = 'flatfiles/all_speakers.csv'

  def extract(self):
    import convokit
    self.corpus = convokit.Corpus(filename="/root/.convokit/downloads/supreme-corpus")
  
  def transform(self):
    self.utterances_df = self.corpus.get_utterances_dataframe().rename_axis('utterance_id')
    conversations_df = self.corpus.get_conversations_dataframe().rename_axis('conversation_id').dropna()
    self.speakers_df = self.corpus.get_speakers_dataframe().rename_axis('speaker_id')
    is_convo_post_1982 = conversations_df['meta.case_id'] \
      .apply(lambda id: int(id.split('_')[0])) >= 1982
    self.conversations_df = conversations_df[is_convo_post_1982]

  def load(self):
    self.utterances_df.to_csv(self.UTTERANCES_PATH)
    self.conversations_df.to_csv(self.CONVERSATIONS_PATH)
    self.speakers_df.to_csv(self.SPEAKERS_PATH)
  

class CreateChunks(Step):
  CHUNKS_PATH = 'flatfiles/chunks.csv'

  def extract(self):
    self.all_utterances = pd.read_csv(ReadCorpus.UTTERANCES_PATH, index_col='utterance_id')
    self.all_conversations = pd.read_csv(ReadCorpus.CONVERSATIONS_PATH, index_col='conversation_id')
    self.all_conversations['meta.advocates'] = self.all_conversations['meta.advocates'].apply(eval)
    self.all_conversations['meta.votes_side'] = self.all_conversations['meta.votes_side'].apply(eval)
    self.all_speakers = pd.read_csv(ReadCorpus.SPEAKERS_PATH)

  def get_chunks_from_utterances(self, convo_id, convo_utts):
    chunks = []
    
    chunk_no = 0
    chunk_start = 0
    
    cur_chunk = []
    cur_advocate = None
    cur_justice = None

    while chunk_start < len(convo_utts):
      # greedily extract chunk
      speaker = convo_utts.iloc[chunk_start]['speaker']
      speaker_type = convo_utts.iloc[chunk_start]['meta.speaker_type']
      
      if speaker_type != 'A':
        # chunks should begin with an advocate speaking
        chunk_start += 1
        continue
      
      # otherwise, begin a new chunk
      cur_advocate = speaker

      chunk_end = chunk_start
      while chunk_end < len(convo_utts):
        utt_id = convo_utts.utterance_id[chunk_end]
        
        speaker2 = convo_utts.iloc[chunk_end]['speaker']
        speaker2_type = convo_utts.iloc[chunk_end]['meta.speaker_type']

        if speaker2_type not in ['A', 'J']:
          # ex. <INAUDIBLE>
          break
        elif speaker2_type == 'A' and speaker2 != cur_advocate:
          # chunks should have only 1 advocate
          break
        elif speaker2_type == 'J' and speaker2 != cur_justice:
          if cur_justice is not None:
            # chunks should have only 1 justice
            break
          else:
            # this is the first time we've encountered a new justice
            cur_justice = speaker2

        cur_chunk.append(utt_id)
        chunk_end += 1
      
      if len(cur_chunk) >= 4 and cur_advocate is not None and cur_justice is not None:
        # chunks must have at least 4 utterances
        chunk_df = pd.DataFrame(dict(
          conversation_id=convo_id,
          chunk_no=chunk_no,
          utterance_id=cur_chunk,
          advocate_id=cur_advocate,
          justice_id=cur_justice
        ))

        merged = chunk_df.merge(convo_utts, on='utterance_id')
        chunk_df['speaker_type'] = merged['meta.speaker_type']
        chunk_df['interrupted'] = merged.text.str[-2:].isin(['..', '--'])
        # a token is matched by the following regex, which matches words (allowing apostrophes) and floating point numbers.
        chunk_df['n_tokens'] = merged.text.str.count(r"([A-Za-z]+('[A-Za-z]+)?)|([-]?([0-9]*[.])?[0-9]+)")

        chunks.append(chunk_df)
        # allow the next chunk to start using the end of this chunk
        chunk_start = chunk_end - 1 
        chunk_no += 1
      else:
        # try to start a chunk at the next utterance
        chunk_start += 1
      
      cur_chunk = []
      cur_advocate = None
      cur_justice = None
    
    if len(chunks) > 0:
      return pd.concat(chunks)
    else:
      return None

  def transform(self):
    dfs = []

    for convo_id in self.all_conversations.index:
      convo_utts = self.all_utterances.query('conversation_id == @convo_id') \
        .reset_index() \
        .rename(columns={'id': 'utterance_id'})
      if (chunks := self.get_chunks_from_utterances(convo_id, convo_utts)) is not None:
        dfs.append(chunks)

    self.chunks = pd.concat(dfs)

  def load(self):
    self.chunks.to_csv(self.CHUNKS_PATH, index=False)
    return dict(chunks=self.chunks)
  

class ComputeInterruptions(Step):
  INTERRUPTIONS_PATH = 'flatfiles/interruptions.csv'

  def extract(self):
    self.chunks = pd.read_csv(CreateChunks.CHUNKS_PATH)
    self.all_conversations = pd.read_csv(ReadCorpus.CONVERSATIONS_PATH)
    self.all_conversations['meta.advocates'] = self.all_conversations['meta.advocates'].apply(eval)
    self.all_conversations['meta.votes_side'] = self.all_conversations['meta.votes_side'].apply(eval)

  def do_align(self, r):
    try:
      return r['meta.advocates'][r.advocate_id]['side'] == r['meta.votes_side'][r.justice_id]
    except:
      return None

  def transform(self):
    all_interruptions = self.chunks.query('speaker_type == "A"') \
      .groupby(['conversation_id', 'chunk_no', 'advocate_id', 'justice_id']).sum()
    interruption_rate = (all_interruptions.interrupted / (all_interruptions.n_tokens / 1000)) \
      .rename('interruption_rate').reset_index()

    votes = self.all_conversations.merge(interruption_rate, on='conversation_id')
    votes['do_align'] = votes.apply(self.do_align, axis=1)

    self.interruptions = votes[interruption_rate.columns.to_list() + ['do_align']].dropna().copy()

  def load(self):
    self.interruptions.to_csv(self.INTERRUPTIONS_PATH, index=False)
    return self.interruptions
  

class ComputeGender(Step):
  GENDER_PATH = 'flatfiles/genders.csv'

  def extract(self):
    self.all_utterances = pd.read_csv(ReadCorpus.UTTERANCES_PATH, index_col='utterance_id')
    self.all_conversations = pd.read_csv(ReadCorpus.CONVERSATIONS_PATH, index_col='conversation_id')
    self.all_conversations['meta.advocates'] = self.all_conversations['meta.advocates'].apply(eval)
    self.all_conversations['meta.votes_side'] = self.all_conversations['meta.votes_side'].apply(eval)
    self.all_speakers = pd.read_csv(ReadCorpus.SPEAKERS_PATH, index_col='speaker_id')

  def transform(self):
    from nameparser import HumanName
    from gender_guesser.detector import Detector as GenderDetector

    all_advocates = self.all_speakers.query('`meta.type` == "A"').rename_axis('advocate_id')
    advocate_names = all_advocates.rename(columns={'meta.name': 'name'})[['name']]
    advocate_names['lastname'] = advocate_names['name'].apply(lambda n: HumanName(n).last)
    advocate_names['male_honorific'] = 'Mr. ' + advocate_names['lastname']
    advocate_names['female_honorific'] = 'Ms. ' + advocate_names['lastname']

    advocates_per_convo = self.all_conversations['meta.advocates'] \
      .apply(lambda a: a.keys()) \
      .explode() \
      .replace('', None) \
      .dropna() \
      .reset_index() \
      .rename(columns={'meta.advocates': 'advocate_id'})
  
    all_honorifics_uttered = self.all_utterances[self.all_utterances.speaker.str.contains('j__')].copy() \
      .set_index('conversation_id')['text'].str.findall(r'(Mr. \w+|Ms. \w+)') \
      .rename('j_honorifics_uttered')
    
    honorifics_uttered = advocates_per_convo.merge(
      all_honorifics_uttered.groupby(level=0).sum().reset_index(), 
      on='conversation_id'
    ).groupby('advocate_id')['j_honorifics_uttered'].sum().apply(set)

    names_and_honorifics = advocate_names.join(honorifics_uttered)
    names_and_honorifics['matched_mr'] = names_and_honorifics.apply(
      lambda a: a['male_honorific'] in a['j_honorifics_uttered'] 
        if type(a['j_honorifics_uttered']) == set else False, 
      axis=1
    )
    names_and_honorifics['matched_ms'] = names_and_honorifics.apply(
      lambda a: a['female_honorific'] in a['j_honorifics_uttered']
        if type(a['j_honorifics_uttered']) == set else False, 
      axis=1
    )

    one_honorific_matched = names_and_honorifics['matched_mr'] ^ names_and_honorifics['matched_ms']
    names_and_honorifics['gender'] = np.where(one_honorific_matched, names_and_honorifics['matched_ms'], None) # female = True, male = False

    # guess missing genders

    detector = GenderDetector()
    names_and_honorifics['firstname'] = names_and_honorifics['name'].apply(lambda n: HumanName(n).first)
    names_and_honorifics['gender_det'] = names_and_honorifics['firstname'].apply(detector.get_gender)

    # set confident detected genders
    names_and_honorifics.loc[names_and_honorifics.gender.isna() & (names_and_honorifics.gender_det == 'male'), 'gender'] = False
    names_and_honorifics.loc[names_and_honorifics.gender.isna() & (names_and_honorifics.gender_det == 'female'), 'gender'] = True
    
    # manually resolve some that are matched with both Mr. and Ms.
    names_and_honorifics.loc[[
      'mary_s_burdick', 
      'gale_norton',
      'corbett_gordon'
    ]] = True

    self.names_and_honorifics = names_and_honorifics

  def load(self):
    self.names_and_honorifics['gender'].to_csv(self.GENDER_PATH)
    return dict(gender=self.names_and_honorifics['gender'])
  

class ComputeExperience(Step):
  EXPERIENCE_PATH = 'flatfiles/experience.csv'

  def extract(self):
    self.all_conversations = pd.read_csv(ReadCorpus.CONVERSATIONS_PATH, index_col='conversation_id')
    self.all_conversations['meta.advocates'] = self.all_conversations['meta.advocates'].apply(eval)
    self.all_conversations['meta.votes_side'] = self.all_conversations['meta.votes_side'].apply(eval)

  def transform(self):
    advocates_per_convo = self.all_conversations['meta.advocates'] \
      .apply(lambda a: a.keys()) \
      .explode() \
      .replace('', None) \
      .dropna() \
      .reset_index() \
      .rename(columns={'meta.advocates': 'advocate_id'})
    self.experience = advocates_per_convo.groupby('advocate_id').count() \
      .rename(columns={'conversation_id': 'experience'})

  def load(self):
    self.experience.to_csv(self.EXPERIENCE_PATH)
    return dict(experience=self.experience)
  

class MergeFields(Step):
  MERGED_FIELDS_PATH = 'flatfiles/merged_fields.csv'

  def extract(self):
    self.genders = pd.read_csv(ComputeGender.GENDER_PATH)
    self.experience = pd.read_csv(ComputeExperience.EXPERIENCE_PATH)
    self.interruptions = pd.read_csv(ComputeInterruptions.INTERRUPTIONS_PATH)

  def transform(self):
    self.merged_fields = self.interruptions.merge(self.genders).merge(self.experience)

  def load(self):
    self.merged_fields.dropna().to_csv(self.MERGED_FIELDS_PATH, index=False)


def run(skip=[]):
  import graph_scheduler
  steps = {
    step_class.__name__: step_class 
    for step_class in [
      ReadCorpus, 
      CreateChunks, 
      ComputeInterruptions, 
      ComputeGender, 
      ComputeExperience,
      MergeFields
    ]
  }

  graph = {
    ReadCorpus: {},
    CreateChunks: {ReadCorpus},
    ComputeInterruptions: {CreateChunks},
    ComputeGender: {ReadCorpus},
    ComputeExperience: {ReadCorpus},
    MergeFields: {ComputeGender, ComputeExperience, ComputeInterruptions}
  }

  stringified_graph = { k.__name__: { v.__name__ for v in graph[k] } for k in graph }
  sched = graph_scheduler.Scheduler(graph=stringified_graph)
  for steps_to_exec in sched.run():
    for step_name in steps_to_exec:
      step = steps[step_name]
      if step not in skip:
        print('Running', step_name)
        step().run()
      else:
        print('Skipping', step_name)

# run(skip=[ReadCorpus, CreateChunks])