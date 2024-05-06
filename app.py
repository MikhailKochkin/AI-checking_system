from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource
from tensor_algo import cosine_distance_with_tensors
from pymystem3 import Mystem
from scipy import spatial
from flask_cors import CORS
from langdetect import detect, detect_langs
from porter2stemmer import Porter2Stemmer
import re


def is_string_empty(s: str) -> bool:
    return len(s) == 0


def createVector(vec, arr, words):
    for i in range(len(words)):
        num = arr.count(words[i])
        vec[i] = num
    return vec


def string_to_list(text):
    words = text.split(" ")
    if len(words) > 1:
        return words
    else:
        return [text]


def preprocess_word(word, raw_text):
    if is_string_empty(raw_text):
        return ""
    lang = detect(raw_text)
    # 1. keep only words
    letters_only_text = word
    # 2. convert to lower case and split
    # Check if only one word, if so return the word as a list
    if len(letters_only_text.split(" ")) == 1:
        letters_only_text = letters_only_text
    else:
        letters_only_text = letters_only_text.lower().split(" ")
    # 3. remove \n
    break_free_words = letters_only_text
    # 5. lemmatize
    lemmatized_words = []
    if (lang == 'ru'):
        m = Mystem()
        if (len(break_free_words.split(" ")) == 1):
            a = m.lemmatize(break_free_words)
            lemmatized_words.append(a[0])
        else:
            for word in break_free_words:
                a = m.lemmatize(word)
                lemmatized_words.append(a[0])
    else:
        if (len(break_free_words.split(" ")) == 1):
            stemmer = Porter2Stemmer()
            lemmatized_words = [stemmer.stem(break_free_words)]
        else:
            stemmer = Porter2Stemmer()
            lemmatized_words = [stemmer.stem(word)
                                for word in break_free_words]

    final = []
    for i in lemmatized_words:
        final.append(i)

    final = [re.sub(r'\[([^][]*)\]', r'[\1]', w) for w in final]
    return final


def preprocess(raw_text):
    if is_string_empty(raw_text):
        return ""
    lang = detect(raw_text)
    # 1. keep only words
    letters_only_text = raw_text
    # 2. convert to lower case and split
    # Check if only one word, if so return the word as a list
    letters_only_text = letters_only_text.lower().split(" ")
    # 3. remove \n
    break_free_words = letters_only_text

    # Remove empty strings from the list
    break_free_words = [word for word in break_free_words if word != ""]
    # [word.rstrip("\n") for word in words]
    # 5. lemmatize
    lemmatized_words = []
    if (lang == 'ru'):
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


def normalize_text(text):
    mapping = {
        "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "у": "y", "х": "x",
        "А": "A", "С": "C", "Е": "E", "О": "O", "Р": "P", "У": "Y", "Х": "X"
    }

    normalized_text = "".join(mapping.get(char, char) for char in text)
    return normalized_text


def is_word_present(word, text):
    word = normalize_text(word.lower())
    text = normalize_text(text.lower())

    return word in text


def is_word_present_strict(word, text):
    word = preprocess_word(normalize_text(word.lower()), text.lower())
    text = preprocess(normalize_text(text.lower()))

    return word[0] in text

def contains_only_numbers(s):
    regex = r'^[0-9]+$'
    return bool(re.match(regex, s))

def compare_strings(str1, str2):
    if len(str1) != len(str2):
        return False

    for i in range(len(str1)):
        if str1[i] != str2[i]:
            return False

    return True


class Checker(Resource):
    def get(self):
        return {"goal": "compare strings"}

    def post(self):
        args = parser.parse_args()
        sample = str(args.answer1)
        answer = str(args.answer2)

        if sample.strip() == "" and answer.strip() == "":
            return {"res": "100", "comment": "", "size_difference_percent": 0}

        if contains_only_numbers(sample) and contains_only_numbers(answer):
            if compare_strings(sample, answer):
                return {"res": "100", "comment": "equal_numbers", "size_difference_percent": 0}
            else:
                return {"res": "10", "comment": "different_numbers", "size_difference_percent": 0}

        answer_preprocessed = answer
        sample_preprocessed = sample

        # Check if a word in square brackets [] in answer1 is missing in answer2
        words_in_brackets = re.findall(
            r'\[(.*?)\]', "".join(sample))

        for word in words_in_brackets:
            if not is_word_present(word, answer):
                return {"res": "10", "comment": "key_info_missing", "size_difference_percent": 0}

         # Check if a word in asterisks ** in answer1 is present in answer2
        words_in_asterisks = re.findall(r'\*(.*?)\*', "".join(sample))

        for word in words_in_asterisks:
            if is_word_present(word, answer):
                return {"res": "10", "comment": "error_words_found", "size_difference_percent": 0}

        for word in words_in_asterisks:
            sample = sample.replace('*' + word + '*', '')

         # Check if a word in angle brackets in answer1 is present in answer2

        words_in_angles = re.findall(r'\<(.*?)\>', "".join(sample))

        for word in words_in_angles:
            if is_word_present_strict(word, answer):
                return {"res": "10", "comment": "error_words_found_strict", "size_difference_percent": 0}

        for word in words_in_angles:
            sample = sample.replace('<' + word + '>', '')

        def extract_misleading_words(text):
            # Define the pattern to match the sentences
            pattern = r'\|([^\|]+)\|'
            # Replace all matches of the pattern in the text with an empty string
            result = re.sub(pattern, '', text)
            return result
        
        def remove_misleading_words_from_answer(text, studentAnswer):
            # Define the pattern to match the sentences
            pattern = r'\|([^|]+)\|'
            
            # Extract the words matching the pattern from the text
            words_to_remove = re.findall(pattern, text)
            
            # Preprocess the words in the words_to_remove list
            preprocessed_words_to_remove = [' '.join(preprocess_word(word, text)) for word in words_to_remove]
            
            # Split the student answer into individual words
            student_answer_words = studentAnswer.split()
            
            modified_student_answer_words = []
            for word in student_answer_words:
                preprocessed_word = ' '.join(preprocess_word(word, studentAnswer))
                if preprocessed_word not in preprocessed_words_to_remove:
                    modified_student_answer_words.append(word)
            
            return ' '.join(modified_student_answer_words)

        answer = remove_misleading_words_from_answer(sample, answer)
        sample = extract_misleading_words(sample)

        answer_split = answer.split(" ")
        filter(lambda x: x != "," or x != "." or x != ":", answer_split)
        sample_split = sample.split(" ")
        filter(lambda x: x != "," or x != "." or x != ":", sample_split)

        if len(answer_split) <= 3 and len(sample_split) <= 3:
            res = compare(answer, sample)
        else:
            res = cosine_distance_with_tensors(answer, sample)
        
        comment = ""

        if res < 65:
            if len(sample_split) / len(answer_split) > 2:
                comment = "more_detailed_response_recommended"
            elif len(answer_split) / len(sample_split) > 2:
                comment = "more_concise_response_recommended"

        sample_words = len(sample.split())
        answer_words = len(answer.split())

        size_difference_percent = round(
            (answer_words - sample_words) /
            max(answer_words, sample_words) * 100
        )

        return {"res": str(res), "comment": comment, "size_difference_percent": size_difference_percent}


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
    app.debug = True
    app.run(host='localhost',port=8000, debug=True)
