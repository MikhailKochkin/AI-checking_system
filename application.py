from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from tensor_algo import cosine_distance_with_tensors
# from glove import compare

application = Flask(__name__)
api = Api(application)

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
        answer = str(args.answer1)
        sample = str(args.answer2)
        answer_split = answer.split(" ")
        filter(lambda x: x != "," or x != "." or x != ":", answer_split)
        sample_split = sample.split(" ")
        filter(lambda x: x != "," or x != "." or x != ":", sample_split)
        # if len(answer_split) <= 3 or len(sample_split) <= 3:
        #     res = compare(answer, sample)
        # else:
        #     res = cosine_distance_with_tensors(answer, sample)

        comment = 0

        res = cosine_distance_with_tensors(answer, sample)


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

if __name__ == "__main__":
    application.run(debug=True)
