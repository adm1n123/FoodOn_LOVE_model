import os
import pandas as pd
from fdc_preprocess import FDCPreprocess
import wikipedia

class Wikipedia:

    def __init__(self):
        self.reuse_summary = True
        self.failed_query_file = 'data/wikipedia/failed_queries.txt'
        self.summary_file = 'data/wikipedia/summaries.txt'
        self.summary_preprocessed_file = 'output/wikipedia_preprocessed.txt'
        self.foodon_pairs_file = 'data/FoodOn/foodonpairs.txt'



    def parse_wiki(self):   # tested
        # take all the labels(vocab) to query wikipedia
        labels = []
        df = pd.read_csv(self.foodon_pairs_file, sep='\t')
        labels.extend(df['Parent'].tolist())
        labels.extend(df['Child'].tolist())
        labels = list(set(labels))

        fdc_preprocess = FDCPreprocess()
        processed_labels = fdc_preprocess.preprocess_columns(pd.Series(labels)).tolist()
        queries = processed_labels.copy()   # queries have all the labels now split each label to get more queries
        for label in processed_labels:
            queries.extend(label.split())
        queries = list(set(queries))

        if self.reuse_summary and os.path.isfile(self.summary_file) and os.path.isfile(self.failed_query_file):
            summary_file = self.summary_file
            failed_query_file = self.failed_query_file
        else:
            summary_file = None
            failed_query_file = None
        summary_df, failed_df = self.query_summary(queries, summary_file, failed_query_file)
        summary_df.to_csv(self.summary_file, sep='\t', index=False)
        failed_df.to_csv(self.failed_query_file, sep='\t', index=False)
        summary_df['summary_preprocessed'] = fdc_preprocess.preprocess_columns(summary_df['summary'], load_phrase_model=True)   # load_phrase_model=True. use phrases for food labels don't make phrases for wiki contents.
        summary_df.to_csv(self.summary_preprocessed_file, sep='\t', index=False)

    def query_summary(self, queries, summary_file=None, failed_query_file=None):    # tested
        if summary_file and failed_query_file:
            print('Reusing previous summaries.')
            pd_prev_summaries = pd.read_csv(summary_file, sep='\t', keep_default_na=False)
            pd_prev_failed = pd.read_csv(failed_query_file, sep='\t', keep_default_na=False)

            known_successful_queries = pd_prev_summaries['query'].tolist()
            known_failed_queries = pd_prev_failed['query'].tolist()
            known_queries = known_successful_queries + known_failed_queries
            depracated_queries = [q for q in known_queries if q not in queries]
            queries = [q for q in queries if q not in known_queries]
            print('Found %d deprecated queries', len(depracated_queries))
            print('Found %d new queries', len(queries))

        summaries = []
        failed_queries = []
        NUM_LOGS = 50
        num_queries = len(queries)
        log_every = [i * int(num_queries / NUM_LOGS) for i in range(NUM_LOGS)]

        for idx, query in enumerate(queries):   # underscores in phrase are treated as space in wiki search.
            if idx in log_every:
                print('Processing query {}/{}'.format(idx, num_queries))
            try:
                summary = wikipedia.WikipediaPage(query).content.replace('\n', ' ')     # to get summary use wikipedia.summary(query)
                summaries.append([query, summary])
            except:
                failed_queries.append([query])

        pd_summaries = pd.DataFrame(summaries, columns=['query', 'summary'])
        pd_failed = pd.DataFrame(failed_queries, columns=['query'])

        if summary_file and failed_query_file:
            pd_summaries = pd_summaries.append(pd_prev_summaries)
            pd_failed = pd_failed.append(pd_prev_failed)
            # pd_summaries = pd_summaries[~pd_summaries['query'].isin(depracated_queries)]  # don't remove any results fetching takes time.
            # pd_failed = pd_failed[~pd_failed['query'].isin(depracated_queries)]
        print('Successfully got wikipedia summaries for %d queries', pd_summaries.shape[0])
        print('Failed to get wikipedia summaries for %d queries', pd_failed.shape[0])
        return pd_summaries, pd_failed




def test():
    # TODO
    # process failed results(it might failed because of some reasons)
    # query phrase with and without underscore and see difference
    return None


if __name__ == '__main__':
    test()