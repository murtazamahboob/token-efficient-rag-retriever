"""Bundled knowledge base: 12 short factual documents.

Fully offline and deterministic. Two photosynthesis documents share a
two-sentence passage verbatim, on purpose: it is the realistic case
(retrieved chunks that restate each other) the deduplicator must catch.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


DOCUMENTS: list[Document] = [
    Document(
        "photosynthesis-basics",
        "Photosynthesis Basics",
        "Photosynthesis is the process by which green plants make food using sunlight, water and carbon dioxide. "
        "Chlorophyll in the leaves captures sunlight and gives plants their green colour. "
        "The plant uses that captured energy to turn water and carbon dioxide into glucose, a sugar it can use for growth. "
        "Oxygen is released as a byproduct of photosynthesis, which is why forests help refresh the air we breathe.",
    ),
    Document(
        "photosynthesis-process",
        "How Photosynthesis Works, Step by Step",
        "Chlorophyll in the leaves captures sunlight and gives plants their green colour. "
        "The plant uses that captured energy to turn water and carbon dioxide into glucose, a sugar it can use for growth. "
        "Light-dependent reactions split water molecules and release oxygen, while the Calvin cycle builds sugars from carbon dioxide. "
        "Photosynthesis runs fastest in bright light, but very high temperatures can damage the enzymes involved.",
    ),
    Document(
        "python-lists",
        "Python Lists",
        "A Python list is an ordered collection that can hold items of different types, such as numbers, strings and other lists. "
        "You create a list with square brackets, for example numbers = [1, 2, 3]. "
        "Lists are mutable, so you can append, insert and remove items after creation. "
        "List comprehensions build a new list from an old one in a single readable line.",
    ),
    Document(
        "python-functions",
        "Python Functions",
        "A Python function is defined with the def keyword, followed by a name and parameters in parentheses. "
        "Functions return a value with the return keyword, and a function without return gives back None. "
        "Arguments are the values you pass when you call a function, and parameters are the names inside its definition. "
        "Default arguments let callers omit values, while keyword arguments make a call self-documenting.",
    ),
    Document(
        "black-holes",
        "Black Holes",
        "A black hole is a region of space where gravity is so strong that nothing, not even light, can escape from it. "
        "The boundary of no return is called the event horizon, and anything crossing it is pulled towards the centre. "
        "At the centre lies the singularity, where matter is crushed to extreme density. "
        "Time runs slower near a black hole, an effect called gravitational time dilation.",
    ),
    Document(
        "vaccines-immune",
        "How Vaccines Work",
        "A vaccine trains the immune system by showing it a harmless form of a pathogen, such as a weakened virus or a single protein. "
        "The immune system responds by producing antibodies that recognise that pathogen. "
        "Memory cells remember the pathogen, so a later real infection is met with a fast, strong response. "
        "This protection lowers the chance of severe illness even when infection still happens.",
    ),
    Document(
        "ocean-currents",
        "Ocean Currents",
        "Ocean currents are large flows of seawater that move heat around the planet. "
        "Wind pushes surface water, while differences in temperature and density drive deeper currents. "
        "Warm currents carry heat from the equator towards the poles, and cold currents bring cold water back. "
        "These currents shape regional climates, fishing grounds and weather patterns.",
    ),
    Document(
        "mars-exploration",
        "Mars Exploration",
        "Mars is the fourth planet from the Sun, a cold desert world with a thin atmosphere. "
        "Rovers such as Perseverance explore its surface, drilling rock samples and searching for signs of ancient water. "
        "Orbiters map the planet and relay data between rovers and Earth. "
        "Future missions aim to return samples to Earth and, eventually, to send astronauts.",
    ),
    Document(
        "neural-networks",
        "Neural Networks",
        "A neural network is a model made of layers of simple units connected by weights. "
        "During training, the network makes predictions, measures its error, and adjusts its weights to reduce that error. "
        "Gradient descent is the algorithm that decides how far to move each weight at every step. "
        "With enough data and training, the layers learn features from simple edges to complex patterns.",
    ),
    Document(
        "databases-indexes",
        "Database Indexes",
        "A database index is a data structure that lets the database find rows without scanning the whole table. "
        "An index on a column speeds up lookups and joins that filter on that column. "
        "Indexes cost extra storage and slow down writes, because every insert must update them. "
        "Choosing the right indexes is a trade-off between read speed and write speed.",
    ),
    Document(
        "climate-greenhouse",
        "The Greenhouse Effect",
        "Greenhouse gases such as carbon dioxide and methane trap heat in the atmosphere. "
        "Sunlight warms the Earth, and greenhouse gases slow the escape of infrared radiation back to space. "
        "Burning fossil fuels has raised carbon dioxide levels and warmed the planet. "
        "Reducing emissions, protecting forests and shifting to clean energy all limit further warming.",
    ),
    Document(
        "bees-pollination",
        "Bees and Pollination",
        "Bees pollinate flowers as they collect nectar, carrying pollen from one flower to another. "
        "Pollination lets plants produce seeds and fruit, including many crops humans eat. "
        "A single bee colony can pollinate millions of flowers in one season. "
        "Protecting bees means protecting wildflower habitat and limiting pesticide use.",
    ),
]


def get_document(doc_id: str) -> Document:
    for doc in DOCUMENTS:
        if doc.doc_id == doc_id:
            return doc
    raise KeyError(f"unknown document: {doc_id}")
