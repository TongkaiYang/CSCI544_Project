from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.websocket import RecognizeCallback, AudioSource

def speechToText(file_path, speech_to_text):
  # open the audio file and run recognition
  audio_file = open(file_path, 'rb')
  content_type = 'audio/' + file_path.split('.')[1]
  response = speech_to_text.recognize(model = 'en-US_BroadbandModel', # english model
                                      audio = audio_file, 
                                      content_type = content_type,
                                      smart_formatting = True, 
                                      profanity_filter = False,
                                      end_of_phrase_silence_time = 0.6
                                     ).get_result()
  audio_file.close()
  return response

def createRecognizer(api_info):
  # setup
  authenticator = IAMAuthenticator(api_info['apikey'])
  speech_to_text = SpeechToTextV1(authenticator=authenticator)
  speech_to_text.set_service_url(api_info['url'])

  return speech_to_text

# sample audio file for test purposes, replace this with actual audio input
file_idx = 1
folder_path = '/data'
sample_list = ['sample1.flac', 'sample2.mp3', 'sample3.flac']

file_path = folder_path + sample_list[file_idx]

# api definition
api_info = {
            "apikey": "pWBFUaj4ktgQ2rXojtRVVUdWvyoFD1HphLVaqOfHDkiB",
            "iam_apikey_description": "Auto-generated for key 5e021eae-f934-40bd-b370-e54abc1656d6",
            "iam_apikey_name": "Auto-generated service credentials",
            "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Manager",
            "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/15ac184c657344819e27057364c1dcb3::serviceid:ServiceId-9e5f761a-bace-4fb0-8465-b2ef8ae046c3",
            "url": "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/2d429bd2-a6dc-4b3d-bb5c-c795d82d400d"
           }

# set up the speech recognizer
speech_to_text = createRecognizer(api_info)

# run speech recognition 
response = speechToText(file_path, speech_to_text)