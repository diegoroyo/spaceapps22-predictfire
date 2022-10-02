import matplotlib.pyplot as plt
import numpy as np
import tifffile
import imageio
import cv2
from tqdm import tqdm


# 8-neighbours
neighbours = [(1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)]

cell_burn_ratio = tifffile.imread('parsed/moncayo_firedanger_yescc_200x200.tif')
RATIO_IN_CELL_DAMPEN = 450
aerial_image = imageio.imread('parsed/moncayo_satellite_500x500.png')
ny, nx = cell_burn_ratio.shape
iy, ix = aerial_image.shape[0:2]

np.random.seed(0)

cell_burned_pct = np.zeros((ny, nx), dtype=np.float32)
cell_candidate_neighbours = np.zeros((ny, nx), dtype=bool)


def start_burn_cell(x, y):
    cell_burned_pct[y, x] = 0.01
    cell_candidate_neighbours[y, x] = False
    for dx, dy in neighbours:
        gx = x + dx
        gy = y + dy
        if gx < 0 or gx >= nx or gy < 0 or gy >= ny:
            continue
        if cell_burned_pct[gy, gx] > 0:
            continue
        cell_candidate_neighbours[gy, gx] = True

hot = plt.get_cmap('hot')

def display():
    ax.cla()
    
    burn_resized = cv2.resize(cell_burned_pct, (ix, iy))

    # non-linearities to make fire colors cool (starts at red, goes to yellow/white, back to red and then to dark red)
    burn_resized_cool = 0.25 + burn_resized * 1.75
    burn_resized_cool[burn_resized_cool < 1.75] /= 1.75
    burn_resized_cool[burn_resized_cool > 1.75] = 1 - ((burn_resized_cool[burn_resized_cool > 1.75] - 1.75) / 0.25)
    burn_resized_cool[burn_resized_cool > 1.75] = 0.15 + (burn_resized_cool[burn_resized_cool > 1.75] * 0.85)
    burn_overlay = hot(burn_resized_cool)[..., 0:3] * 255

    aerial_image_show = np.copy(aerial_image)
    aerial_image_show[burn_resized > 0] = burn_overlay[burn_resized > 0]
    mixed = np.logical_and(burn_resized > 0, burn_resized < 0.5)
    mix_1 = aerial_image[mixed] * np.stack((1 - 2 * burn_resized[mixed],)*3, axis=-1)
    mix_2 = burn_overlay[mixed] * np.stack((2 * burn_resized[mixed],)*3, axis=-1)
    aerial_image_show[mixed] = mix_1 + mix_2

    ax.imshow(aerial_image_show)
    fig.savefig(f'images/moncayo_bottomright_yescc/{total_i:04d}.png', bbox_inches='tight', pad_inches=0)

    plt.draw()


def update():
    # Update burning cells
    burning_cells = np.logical_and(0 < cell_burned_pct, cell_burned_pct < 1)
    cell_burned_pct[burning_cells] = np.minimum(
        1.0, cell_burned_pct[burning_cells] + cell_burn_ratio[burning_cells] / RATIO_IN_CELL_DAMPEN)

    # Check candidates for burning
    cell_candidate_neighbour_ids = np.where(cell_candidate_neighbours)
    for y, x in zip(*cell_candidate_neighbour_ids):
        # Check the burned ptc of the neighbors
        x0 = max(0, x - 1)
        y0 = max(0, y - 1)
        x1 = min(nx - 1, x + 1)
        y1 = min(ny - 1, y + 1)
        window = cell_burned_pct[y0:y1+1, x0:x1+1]
        cell_burned_pct_pool = np.max(window[window < 1]) # Take the max that is not calcinated
        # Combine probabilities
        burn_prob = (cell_burned_pct_pool * cell_burn_ratio[y, x] ** 0.9)
        # Check if the cell will be burned
        if np.random.uniform() < burn_prob:
            start_burn_cell(x, y)


if __name__ == '__main__':
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ax.imshow(aerial_image)
    # x, y = fig.ginput(1, timeout=0)[0]
    # # convert from image range to grid range
    # x = int(x * nx / ix)
    # y = int(y * ny / iy)
    # start_burn_cell(x, y)
    # plt.close()
    # for test1 experiments
    # start_burn_cell(int(143.0 * 200 / 500), int(249.0 * 200 / 500))
    # for test2 experiments
    start_burn_cell(int(441.0 * 200 / 500), int(239.0 * 200 / 500))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.ion()

    import time
    N_FRAMES = 150
    total_i = 0
    total_t_display = 0
    total_t_update = 0
    with tqdm(desc='Simulating...', total=N_FRAMES) as pbar:
        while total_i < N_FRAMES:
            time_a = time.time()
            display()
            time_b = time.time()
            for _ in range(60):
                update()
            time_c = time.time()
            # print(np.sum(cell_candidate_neighbours))
            plt.pause(0.001)
            total_i += 1
            total_t_display += time_b - time_a
            total_t_update += time_c - time_b
            pbar.desc = f'Mean t_display = {total_t_display / total_i:.3f} s | Mean t_update = {total_t_update / total_i:.3f} s)'
            pbar.update(1)