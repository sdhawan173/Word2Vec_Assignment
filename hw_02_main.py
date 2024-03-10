import text_functions as tfx
import text_file_operations as tfo

print('----------PART 1----------')
positive_data = tfo.load_data('comments1k_pos')
negative_data = tfo.load_data('comments1k_neg')
all_data = positive_data + negative_data
data_split_words = tfx.split_words(all_data, exclusion=True)
print('PART 1, Question 1--------')
word2vec_model = tfx.word2vec_cbow(data_split_words, vector_size=100, window=5, min_count=1)
tfx.vector_visualize(word2vec_model, word2vec=True)
print('PART 1, Question 2--------')
glove_model = tfx.glove_world(data_split_words)
tfx.vector_visualize(glove_model, glove=True)
print('PART 1, Question 3--------')
eval_list = ['movie', 'music', 'woman', 'Christmas']
tfx.model_eval(eval_list, 10, word2vec_model=word2vec_model)
tfx.model_eval(eval_list, 10, glove_model=glove_model)
print('PART 1, Question 4--------')
word2vec_m1 = tfx.word2vec_cbow(data_split_words, vector_size=100, window=5, min_count=1)
word2vec_m2 = tfx.word2vec_cbow(data_split_words, vector_size=100, window=5, min_count=10)
word2vec_m3 = tfx.word2vec_cbow(data_split_words, vector_size=100, window=5, min_count=100)

print('PART 1, Question 5--------')

