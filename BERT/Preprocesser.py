from underthesea import word_tokenize


class Preprocesser:
    def __init__(self, sw_path):
        self.stopwords_path = sw_path

    def fill_null(self, df):
        df = df.fillna("")
        return df

    def concat_summary_and_content(self, df):
        list_1 = df['summary'].to_list()
        list_2 = df['content'].to_list()
        list_text = []
        for i in range(len(list_1)):
            list_text.append(list_1[i] + ' ' + str(list_2[i]))

        df['text'] = list_text
        df['text'] = df['text'].str.lower()

        return df

    # list_text = df['text'].to_list()
    # #print(list_text[0])
    # for i in range(len(list_text)):
    #     list_text[i] = word_tokenize(list_text[i], format='text')

    # df['text'] =  list_text
    def tokenizer_cc(self, df):
        list_text = df['text'].to_list()

        for i in range(len(list_text)):
            list_text[i] = word_tokenize(list_text[i], format='text')

        df['text'] = list_text

        return df

    def get_stopwords_list(self, stop_file_path):
        """load stop words """

        with open(stop_file_path, 'r', encoding="utf-8") as f:
            stopwords = f.readlines()
            stop_set = set(m.strip() for m in stopwords)
            return list(frozenset(stop_set))

    def remove_stop_words(self, corpus):
        results = []
        stopwords = self.get_stopwords_list(self.stopwords_path)
        for text in corpus:
            tmp = text.split(' ')
            for stop_word in stopwords:
                if stop_word in tmp:
                    tmp.remove(stop_word)
            results.append(" ".join(tmp))

        return results

    def processing(self, df):
        df = self.fill_null(df)
        df = self.concat_summary_and_content(df)
        df = self.tokenizer_cc(df)
        stopwords = self.get_stopwords_list(self.stopwords_path)
        df['text'] = self.remove_stop_words(df['text'].to_list())
        return df