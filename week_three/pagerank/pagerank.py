import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    pd = {}
    total_pages = len(corpus)

    links = corpus.get(page, set())

    if not links:
        for p in corpus:
            pd[p] = 1 / total_pages
    else:
        link_prop = damping_factor / len(links)
        random_prop = round((1 - damping_factor) / total_pages, 4)
        for p in corpus:
            pd[p] = random_prop
            if p in links:
                pd[p] += link_prop
    return pd


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    spr = {page: 0 for page in corpus}

    page = random.choice(list(spr.keys()))

    for _ in range(n):
        spr[page] += 1
        pd = transition_model(corpus, page, damping_factor)
        page = random.choices(list(pd.keys()), weights=pd.values(), k=1)[0]

    for p in spr:
        spr[p] /= n
    return spr


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ipr = {page: 1 / len(corpus) for page in corpus}
    new_ipr = ipr.copy()

    while True:
        for page in ipr:
            new_ipr[page] = (1 - damping_factor) / len(corpus)
            for p in corpus:
                if page in corpus[p]:
                    new_ipr[page] += damping_factor * ipr[p] / len(corpus[p])

        if all(abs(ipr[page] - new_ipr[page]) < 0.001 for page in ipr):
            break
        ipr = new_ipr.copy()

    return ipr


if __name__ == "__main__":
    main()
