from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from tensor_algo import cosine_distance_with_tensors
from pymystem3 import Mystem
from scipy import spatial
from flask_cors import CORS
from langdetect import detect
from porter2stemmer import Porter2Stemmer


def createVector(vec, arr, words):
    for i in range(len(words)):
        num = arr.count(words[i])
        vec[i] = num
    return vec

def preprocess(raw_text):
    lang = detect(raw_text)
    # 1. keep only words
    letters_only_text = raw_text
    # 2. convert to lower case and split
    words = letters_only_text.lower().split()
    # 3. remove \n
    break_free_words = words
    # [word.rstrip("\n") for word in words]
    # 5. lemmatize
    lemmatized_words = []
    print(break_free_words)
    if(lang == 'ru'):
        m = Mystem()
        for word in break_free_words:
            a = m.lemmatize(word) 
            lemmatized_words.append(a[0])
    else:
        stemmer = Porter2Stemmer()
        lemmatized_words = [stemmer.stem(word) for word in break_free_words]
    final = []
    for i in lemmatized_words:
        final.append(i)
    return final

def compare(result, model):
    result = preprocess(result)
    model = preprocess(model)
    all_words_in_sentences = result + model
    all_words_in_sentences = list(set(all_words_in_sentences))
    vec1_empty = [None] * len(all_words_in_sentences)
    vec2_empty = [None] * len(all_words_in_sentences)

    vector_1 = createVector(vec1_empty, result, all_words_in_sentences)
    vector_2 = createVector(vec2_empty, model, all_words_in_sentences)

    cosine = spatial.distance.cosine(vector_1, vector_2)

    return round((1 - cosine) * 100, 2)

app = Flask(__name__)
CORS(app)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument("answer1")
parser.add_argument("answer2")

class Index(Resource):
    def get(self):
        return {"start": "page"}

class Checker(Resource):
    def get(self):
        return {"goal": "compare strings"}

    def post(self):
        args = parser.parse_args()
        sample = str(args.answer1)
        answer = str(args.answer2)
        answer_split = answer.split(" ")
        filter(lambda x: x != "," or x != "." or x != ":", answer_split)
        sample_split = sample.split(" ")
        filter(lambda x: x != "," or x != "." or x != ":", sample_split)
        if len(answer_split) <= 3 and len(sample_split) <= 3:
            print(1)
            res = compare(answer, sample)
        else:
            res = cosine_distance_with_tensors(answer, sample)
            print(2)

        comment = 0

        if res < 70:
            if len(sample_split) / len(answer_split) > 2:
                comment = "Дайте более развернутый ответ"
            elif len(answer_split) / len(sample_split) > 2:
                comment = "Дайте более короткий ответ"

        return {"res": str(res), "comment": comment}


##
# Actually setup the Api resource routing here
##
api.add_resource(Index, "/")
api.add_resource(Checker, "/checker")

# if __name__ == "__main__":
#     application.run(debug=True)

if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    # application.debug = True
    app.run()
