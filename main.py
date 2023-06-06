from google.cloud import texttospeech
import speech_recognition as sr
from langchain.agents import ConversationalAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import SerpAPIWrapper

import vlc

import os
from dotenv import load_dotenv, find_dotenv


def initialize(verbose_agent=False):
    load_dotenv(find_dotenv())

    SERPAPI_API_KEY = os.environ["SERPAPI_API_KEY"]
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    search = Tool(
        name="Current Search",
        func=SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY).run,
        description="useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term."
    )

    tools = [
        search
    ]

    prefix = """
        You are a smart home assitant named JARVIS. Have a conversation with a human that goes by the name {userName}, answering the following questions as best you can. You have access to the following tools:
    """

    suffix = """
        {chat_history}
        {userName} (Human): {input}
        JARVIS (You): {agent_scratchpad}
    """

    prompt = ConversationalAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["userName", "input",
                         "chat_history", "agent_scratchpad"]
    ).partial(userName="Aviral Dhingra")

    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(llm=OpenAI(
        temperature=0.2, openai_api_key=OPENAI_API_KEY), prompt=prompt)
    agent = ConversationalAgent(
        llm_chain=llm_chain, tools=tools, verbose=verbose_agent)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=verbose_agent, memory=memory)
    return agent_chain


def speak(res):
    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=res)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary.
    with open("output.mp3", "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

    vlc.MediaPlayer("output.mp3").play()


def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:

        # initialize the recognizer
        # read the audio data from the default microphone
        audio_data = r.record(source, duration=5)
        print("Recognizing...")
        # convert speech to text
        text = r.recognize_google(audio_data)
        print(text)
        return text


if __name__ == "__main__":
    JARVIS = initialize()

    status = True

    while status:
        try:
            query = listen()
            res = f"JARVIS: {JARVIS.run(input=query)}\n"
            print(res)
            speak(res.replace("JARVIS: ", ""))
        except sr.exceptions.UnknownValueError:
            pass
        except KeyboardInterrupt:
            print("\n\n[+] Exiting...")
            status = False
            break
