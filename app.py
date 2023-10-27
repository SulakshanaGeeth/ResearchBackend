import subprocess
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
import uuid
import fasttext
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datasets import Dataset, Audio
from pyannote.audio import Model, Inference
from transformers import WhisperProcessor, WhisperForConditionalGeneration
warnings.filterwarnings("ignore")


app = Flask(__name__)
cors = CORS(app)

audio_embedding_model = Model.from_pretrained(
    "pyannote/embedding",
    use_auth_token="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"
)
audio_embedding_inference = Inference(
    audio_embedding_model,
    window="whole"
)

s2t_processor = WhisperProcessor.from_pretrained(
    "Subhaka/whisper-small-Sinhala-Fine_Tune")
s2t_model = WhisperForConditionalGeneration.from_pretrained(
    "Subhaka/whisper-small-Sinhala-Fine_Tune")
s2t_forced_decoder_ids = s2t_processor.get_decoder_prompt_ids(
    language="sinhala",
    task="transcribe"
)
sentence_embedding_model = fasttext.load_model("models/cc.si.300.bin")


class_dict_abnomalities = {
    'autism': 0,
    'non-autism': 1
}
class_dict_abnomalities_rev = {v: k for k,
                               v in class_dict_abnomalities.items()}


model_abnomalities = tf.keras.models.load_model(
    'models/abnomility-sentiment.h5')
model_answerability = tf.keras.models.load_model(
    'models/answering-evaluation.h5')
model_pronounce = tf.keras.models.load_model('models/pronounce-validation.h5')


def inference_abnomility_sentiment(audio_file):

    embedding = audio_embedding_inference(audio_file)
    embedding = np.expand_dims(embedding, axis=0)
    sentiment = model_abnomalities.predict(embedding)
    sentiment = sentiment.squeeze()
    sentiment = np.round(sentiment)
    sentiment = int(sentiment)
    return class_dict_abnomalities_rev[sentiment]


def load_audio(audio_file):

    audio_data = Dataset.from_dict(
        {"audio": [audio_file]}
    ).cast_column("audio", Audio())
    audio_data = audio_data.cast_column(
        "audio",
        Audio(sampling_rate=16000)
    )
    audio_data = audio_data[0]['audio']['array']
    return audio_data


def transcribe(audio_file):
    audio_data = load_audio(audio_file)
    input_features = s2t_processor(
        audio_data,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features
    predicted_ids = s2t_model.generate(
        input_features,
        forced_decoder_ids=s2t_forced_decoder_ids
    )

    # transcription = s2t_processor.batch_decode(predicted_ids)
    transcription = s2t_processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )
    return transcription[0]


def inference_answer_evaluation(
    audio_file01,
    audio_file02
):
    transcription_01 = transcribe(audio_file01)
    transcription_02 = transcribe(audio_file02)

    embedding01 = sentence_embedding_model.get_sentence_vector(
        transcription_01)
    embedding02 = sentence_embedding_model.get_sentence_vector(
        transcription_02)

    embedding01 = np.expand_dims(embedding01, axis=0)
    embedding02 = np.expand_dims(embedding02, axis=0)

    prediction = model_answerability.predict([embedding01, embedding02])
    prediction = prediction.squeeze()

    prediction = np.round(prediction)
    prediction = int(prediction)
    return 'non-autism' if prediction == 1 else 'autism'


def inference_pronounce_validation(
    audio_file01,
    audio_file02
):
    embedding01 = audio_embedding_inference(audio_file01)
    embedding02 = audio_embedding_inference(audio_file02)

    embedding01 = np.expand_dims(embedding01, axis=0)
    embedding02 = np.expand_dims(embedding02, axis=0)

    prediction = model_pronounce.predict([embedding01, embedding02])
    prediction = prediction.squeeze()

    prediction = np.round(prediction)
    prediction = int(prediction)

    return 'non-autism' if prediction == 1 else 'autism'


@app.route('/abnomility-sentiment', methods=['POST'])
def abnomility_sentiment():
    if request.method == 'POST':
        print("Component 01 request.data----:", request.files)
        audio_file = request.files['audio']

        aduio_save_path = f'uploads/abnomility-sentiment/{uuid.uuid4()}.wav'
        aduio_save_path2 = f'uploads/abnomility-sentiment/{uuid.uuid4()}.wav'
        audio_file.save(aduio_save_path)

        subprocess.run(["ffmpeg", "-i", aduio_save_path,
                       f"{aduio_save_path2}"])

        print('aduio_save_path -----', aduio_save_path)

        sentiment = inference_abnomility_sentiment(
            f"{aduio_save_path2}")
        print(sentiment)
        return jsonify({'abnomility-sentiment': sentiment}), 200

    return jsonify({'status': 'error'}), 400


@app.route('/answer-evaluation', methods=['POST'])
def answer_evaluation():
    if request.method == 'POST':

        audio_file01 = request.files['files01']
        audio_file02 = request.files['files02']

        aduio_save_path01 = f'uploads/answer-evaluation/{uuid.uuid4()}.wav'
        aduio_save_path02 = f'uploads/answer-evaluation/{uuid.uuid4()}.wav'

        aduio_save_path03 = f'uploads/answer-evaluation/{uuid.uuid4()}.wav'
        aduio_save_path04 = f'uploads/answer-evaluation/{uuid.uuid4()}.wav'

        audio_file01.save(aduio_save_path01)
        audio_file02.save(aduio_save_path02)

        subprocess.run(["ffmpeg", "-i", aduio_save_path01,
                        f"{aduio_save_path03}"])
        subprocess.run(["ffmpeg", "-i", aduio_save_path02,
                        f"{aduio_save_path04}"])

        sentiment = inference_answer_evaluation(
            aduio_save_path03, aduio_save_path04)
        return jsonify({'answer-evaluation': sentiment}), 200

    return jsonify({'status': 'error'}), 400


@app.route('/pronounce-validation', methods=['POST'])
def pronounce_validation():
    if request.method == 'POST':

        print("Component 01 request.data----:", request.files)
        audio_file01 = request.files['files01']
        audio_file02 = request.files['files02']

        aduio_save_path01 = f'uploads/pronounce-validation/{uuid.uuid4()}.wav'
        aduio_save_path02 = f'uploads/pronounce-validation/{uuid.uuid4()}.wav'

        aduio_save_path03 = f'uploads/pronounce-validation/{uuid.uuid4()}.wav'
        aduio_save_path04 = f'uploads/pronounce-validation/{uuid.uuid4()}.wav'

        audio_file01.save(aduio_save_path01)
        audio_file02.save(aduio_save_path02)

        subprocess.run(["ffmpeg", "-i", aduio_save_path01,
                        f"{aduio_save_path03}"])
        subprocess.run(["ffmpeg", "-i", aduio_save_path02,
                        f"{aduio_save_path04}"])

        sentiment = inference_pronounce_validation(
            aduio_save_path03, aduio_save_path04)
        return jsonify({'pronounce-validation': sentiment}), 200

    return jsonify({'status': 'error'}), 400


if __name__ == '__main__':
    app.run(debug=True)
