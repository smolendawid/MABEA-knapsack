import json

from sklearn.manifold import TSNE
import plotly.express as px


if __name__ == '__main__':
    path = 'results/05-Dec-2020-01-13-14/genotypes_190.json'

    genotypes = []
    with open(path) as f:
        lines = f.read().splitlines()
        for line in lines:
            genotype = json.loads(",".join(line.split(',')[:-1]))
            genotypes.append(genotype)

    perplexity = 20
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    feats2d = tsne.fit_transform(genotypes)

    data = {'x': feats2d[:, 0], 'y': feats2d[:, 1]}
    fig = px.scatter(data, x='x', y='y', title=f'TSNE 2d projection of genotypes from<br>{path}');fig.show()
    fig.write_html(f"data/2d_{perplexity}.html")
