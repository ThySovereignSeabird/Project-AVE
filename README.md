# Project-AVE

Objective 0: Build an algorithm to derive the K-color palette hierarchy of a sprite in the form of a K-by-K matrix.

Objective 1: Build an interpretable critic for 16×16 item textures that estimates stylistic and volumetric consistency using domain-informed representations.

## Papers:
- Pixel VQ-VAEs for Improved Pixel Art Representation (https://arxiv.org/abs/2203.12130)
- Generating Gameplay-Relevant Art Assets with Transfer Learning (https://arxiv.org/pdf/2010.01681)

## Color ramps
Björn Ottosson, the creator of the Oklab color space, designed it specifically so that a linear interpolation (a straight vector with a constant direction) maintains a visually uniform hue and a linear progression of lightness (Ottosson, 2020).

https://opengameart.org/forumtopic/some-insights-about-pixel-art-color-ramps

The process of constructing a color ramp is as follows:

- A dict is constructed to map every unique color in the sprite to a list of the positions of all of pixels of that color.
    - The keys of this dict are the list of unique colors in the sprite, including the "transparent" color.

- An adjacency matrix (2D array) tracks whether color i is adjacent to color j in the sprite at least once. This lends credence for ranpiness but is not 
- A distance matrix (2D array), initialized with np.inf, memoizes Oklab distances (floats) between colors as they are computed.

- Let "ramp matrix" where entry [i][j] expresses the confidence that color i is succeeded by color j in the sprite's color ramp.
- Rampiness (likelihood function justifying connection between colors i and j):
    - Note that colors i and j are monotonically descending in L.
    - Penalize |ΔL| > constant.
    - Penalize (ΔL / distance) < constant.
    - 

- I want to account for cosine similarity between vectors and the distance between points


OLD:
- A dict is constructed to map every unique color in the sprite to a list of the positions of all of pixels of that color.
    - For every pixel of a color i, add adjacency to the colors of its cardinal neighbors as well as its diagonal neighbors (weighted differently).
    - Adjacency should be deducted for colors with too small or too large absolute ΔL. This is likely the factor that makes or breaks adjacency.
        - Color ramps should monotonically descend in lightness (highest to lowest); there should be a harsh penalty for nonpositive ΔL.
    - Adjacency should be proportionally deducted (to a lesser extent) for colors with excess ΔL relative to the neighbor with the smallest ΔL.
    - Adjacency should be proportionally deducted (to an even lesser extent) for colors with excess ΔC = sqrt(a^2 + b^2) relative to the neighbor with the smallest ΔC.
        - Namely, if the arrow would be too horizontal given the extent of drop in ΔL. Consider computing cosine similarity to chroma axis vectors.
    - When all neighbors have been processed:
    - Repeat the algorithm: 


## Contrast

### Principles of figure-ground color perception:
- Different-hue grounds leech the same hue from the figure
- Higher-value grounds decrease the figure value, lower-value grounds boost the figure value
- Higher-intensity grounds decrease the figure intensity, lower/no-intensity grounds boost the figure intensity

### The Scientific Concept: The Vision Pathways
Human vision splits processing into two streams:
- The magnocellular pathway: Fast, high spatial resolution, handles high-detail edges and contrast. It is completely color-blind and reads only Lightness (L).
- The parvocellular pathway: Slower, low spatial resolution, handles color fields (a,b). It blurs fine text and one-pixel borders.
When a sprite is small, the parvocellular system drops its acuity dramatically. The brain stops registering the precise a and b distance and relies almost entirely on the L channel to resolve the asset's shape.

### Perceptual contrast
ΔE = 0.02 is a standard minimum distance for color distinguishability, but perception under certain conditions can piece out ΔE = 0.0077 and under

Perceptual contrast in pixel art can be computed from a weighted distance formula of L, a, b in the Oklab color space:
- sqrt(w_1(L1 - L2)^2 + w_2(a1 - a2)^2 + w_3(b1 - b2)^2)

Outlines:
- Softness: Delta L is too small
- Harshness: Delta L is too large

Shading:
- Flatness: Delta L is too small
- Muddiness: Delta L is too small and Delta a or b is too large
- Harshness: Delta L is too large

Highlights:
- Chalkiness: Delta L is too large
- Garishness: Delta a or b is too large