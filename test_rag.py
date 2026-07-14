"""
Standalone RAG test - run this BEFORE touching the voice agent.

Confirms retrieval returns correct, restaurant-specific results and
never leaks one restaurant's info when querying the other.
"""

from rag.vector_store import search_restaurant

RESTAURANTS = ["bundu_khan", "cafe_aylanto"]


def run_query(restaurant_id: str, query: str):
    print(f"\n{'=' * 60}")
    print(f"Restaurant: {restaurant_id}")
    print(f"Query: {query}")
    print("-" * 60)
    result = search_restaurant(restaurant_id, query, top_k=3)
    print(result)


if __name__ == "__main__":
    # Test 1: same query across both restaurants - results should differ
    run_query("bundu_khan", "do you have pizza")
    run_query("cafe_aylanto", "do you have pizza")

    # Test 2: menu-specific query
    run_query("bundu_khan", "what mutton dishes do you have")

    # Test 3: policy query
    run_query("bundu_khan", "what is your reservation policy")
    run_query("cafe_aylanto", "how long does takeaway take")

    # Test 4: price query
    run_query("cafe_aylanto", "how much is the margherita pizza")

    print(f"\n{'=' * 60}")
    print("If each result only shows that restaurant's own items/policy,")
    print("with no cross-contamination between the two, RAG is working.")