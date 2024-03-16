from pathlib import Path

import streamlit as st
import wikipedia
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.agent import ReActAgent
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import TextNode
from llama_index.core.tools import RetrieverTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.readers.web import SimpleWebPageReader
from pydantic import BaseModel

wikipedia.set_lang("cs")

system_prompt = "You are amazing trip planner."

llm_4_turbo = OpenAI(model="gpt-4-turbo-preview", temperature=0, system_prompt=system_prompt, timeout=120)

cities = {
    "Brno": {
        "wiki_pages": [
            "Brno",
            "Pivovar Starobrno"
        ],
        "web_pages": [
            "https://mendelmuseum.muni.cz/o-muzeu/mendelovo-muzeum",
            "https://www.tmbrno.cz/vystavy-a-akce/vystavy/",
            "https://www.alkoholium.cz/nejlepsi-bary-v-brne/",
        ]
    },
}

group_types = [
            "Family with Kids",
            "Group of Python Developers",
            "Bachelor party"
        ]


class Attraction(BaseModel):
    """
    Attractions person can visit on a trip to given city.
    Name must be exact name of a place person can visit. For example Mendelovo muzeum, Cukrárna BezCukru
    Description should be a long text
    """
    name: str
    description: str


class AttractionsInCity(BaseModel):
    """List of attractions person can visit on a trip to given city"""
    attractions: list[Attraction]


class ItineraryStop(BaseModel):
    """
    Itinerary stop on a trip to given city.
    time_arrival and time_end MUST be in 24-hour format.
    name and description should contain emojis.
    """

    time_arrival: str
    time_end: str
    name: str
    description: str


class TravelItinerary(BaseModel):
    """Itinerary when visiting a given city"""

    stops: list[ItineraryStop]


def gen_attractions(text: str) -> list[Attraction]:
    summarizer = TreeSummarize(output_cls=AttractionsInCity, llm=llm_4_turbo)
    places_in_city = summarizer.get_response(
        "List separate attractions a person can enjoy when visiting the city, including pubs, restaurants, parks, and more.", [text]
    )

    return places_in_city.attractions


def get_attractions(city: str):
    all_attractions = []

    wiki_pages = cities[city]["wiki_pages"]
    web_pages = cities[city]["web_pages"]

    for wiki_page in wiki_pages:
        print(f"Processing Wiki {wiki_page}")
        wiki_page_content = wikipedia.page(wiki_page).content
        all_attractions += gen_attractions(wiki_page_content)

    for web_page in web_pages:
        print(f"Processing Web {web_page}")
        web_page_content = SimpleWebPageReader(html_to_text=True).load_data(
            [web_page]
        )[0].text
        all_attractions += gen_attractions(web_page_content)

    return all_attractions


def generate_itinerary(city: str, group_type, start_time, end_time):
    print(city, group_type, start_time, end_time)

    index_dir_name = Path(f"./index_{city.lower()}/")
    if index_dir_name.exists():
        storage_context = StorageContext.from_defaults(persist_dir=index_dir_name)

        places_index = load_index_from_storage(storage_context)
    else:
        all_attractions = get_attractions(city)

        places_nodes = [TextNode(text=f"{p.name} - {p.description}") for p in all_attractions]
        places_index = VectorStoreIndex(places_nodes)

        places_index.storage_context.persist(persist_dir=index_dir_name)

    query_engine_tools = [
        RetrieverTool(
            places_index.as_retriever(similarity_top_k=3),
            metadata=ToolMetadata(
                name="city_places_list",
                description=(
                    "Searches interesting places in city based on place's description."
                    "Use detailed description what places are you looking for with examples."
                    "Describe if you are looking for restaurants, museums etc."
                ),
            ),
        ),
    ]

    agent_prompt = f"""
    Generate complete travel itinerary, starting at {start_time} and ending at {end_time} to a single day trip to {city}.
    Trip is prepared for {group_type}.
    Divide day into multiple parts and use tools for each of them separately.
    Think about attractions and places to eat.
    Mention concrete attractions you find using tools.
    Ensure activities start at {start_time}.
    Ensure activities are planned until {end_time}.
    """

    agent = ReActAgent.from_tools(query_engine_tools, llm=llm_4_turbo, verbose=True, max_iterations=20)
    agent_response = agent.chat(agent_prompt).response

    summarizer = TreeSummarize(output_cls=TravelItinerary, llm=llm_4_turbo)
    response = summarizer.get_response("Parse travel itinerary", [agent_response])

    print(agent_response)

    return response.stops


def main():
    st.title("City Trip Planner")

    city = st.selectbox(
        "Choose the city for your trip:",
        cities.keys(),
        index=0
    )

    group_type = st.radio(
        "Who is traveling?",
        group_types
    )

    start_time = st.selectbox(
        "Select the start time of your visit:",
        [f"{i}:00" for i in range(24)],  # 0:00 to 23:00
        index=8
    )

    end_time = st.selectbox(
        "Select the end time of your visit:",
        [f"{i}:00" for i in range(24)],
        index=17
    )
    if st.button('Generate Trip Itinerary'):
        itinerary = generate_itinerary(city, group_type, start_time, end_time)

        st.subheader(f'Travel Itinerary for *{group_type}* to *{city}* ', divider='rainbow')
        for item in itinerary:
            st.text_area(
                f"{item.time_arrival} - {item.time_end}: **{item.name}**",
                item.description
            )


if __name__ == "__main__":
    main()
