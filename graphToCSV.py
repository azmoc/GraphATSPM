"""
Darian Comsa
This is an experiment to see if I can extract data points from 
ATSPM volume approach graphs.
Convert the jpeg to rgb and then find the bit in that column
"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import sys


def GetBox(color_band):
    """Find the graph boundaries"""
    row_vals = np.sort(np.sum(color_band, axis=1))
    num_rows = np.count_nonzero(row_vals < 150_000)
    print(f'num rows: {num_rows}')
    print(row_vals[:num_rows + 1])
    row_grid_inds = np.sort(np.argsort(np.sum(color_band, axis=1))[:num_rows])
    col_grid_inds = np.sort(np.argsort(np.sum(color_band, axis=0))[:25])
    return row_grid_inds, col_grid_inds, num_rows - 1

def GetGraphVals(color_band, row_grid_inds, col_grid_inds, is_red):
    """Reads graph points"""
    vals = []
    # Space them properly but the last value is not graphed
    # box_left = col_grid_inds[0]
    # box_right = col_grid_inds[-1]
    box_top = row_grid_inds[0]
    box_bot = row_grid_inds[-1]
    height = box_bot - box_top
    shift_row_inds = row_grid_inds - box_top
    quarters = []
    for i,c in enumerate(col_grid_inds[:-1]):
        # Divide each grid square into 4 columns to sample
        quarters += np.linspace(c, col_grid_inds[i+1], 5)[:-1].astype(int).tolist()
    quarters = np.array(quarters)
    quarters[::4] += 1
    quarters[-1] -= 1
    for n,i_int in enumerate(quarters):
        # Looking at this column and one pixel right
        pixel_columns = color_band[box_top: box_bot + 3, i_int: i_int + 2].astype(float)[::-1]
        # if not is_red and 90 < n < 92:
        #     print(pixel_columns)
        #     print(pixel_columns[height - shift_row_inds + 2])
        # Clean out the grid lines
        # k = np.where(k > 1, k, 255)
        pixel_columns[height - shift_row_inds + 2] = 230
        # Average between this pixel and the next one up
        k_av = np.array([np.sum(pixel_columns[pix:pix + 2]/2) for pix in range(height)])[:-1]
        # if np.argmin(k_av) > height-2:
        #     print(k[-40:])
        #     print(k_av[-40:])
        vals.append(np.argmin(k_av))
    return np.array(vals)


if __name__ == "__main__":
    show_graphs = False
    file_names = []
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        # print(sys.argv)
        print(f"\nArguments of the script : {args}")
        if 'True' in args:
            show_graphs = True
            args.remove('True')
        file_names = args
    if not len(file_names):
        MONTH_DAY_COUNTS = [31,29,31,30,31,30,31,31,30,31,30,31]
        # file_name = r"Graphs\VolumesGraph2020-05-28.jpg"
        # file_name = r"Graphs\testVolumesGraph.jpg"
        signalID = '6415'
        file_base = "GraphsPt2\\VolumesGraphPt2_2020-01-"
        month_days = MONTH_DAY_COUNTS[0]
        file_names = [file_base + f"{d+1:02d}.jpg" for d in range(month_days)]
        # file_names = ["Graphs\\testVolumes2019-10-01.jpg"]
    # NUM_ROWS = 8
    # Problem graphs
    north_v = []
    south_v = []
    x_scaled = np.linspace(0, 24, 97)
    for name in file_names[:]:
        print(f'\n{name}')
        im = Image.open(name)
        rim, gim, bim = im.split()
        
        # https://stackoverflow.com/questions/10825217/edit-rgb-values-in-a-jpg-with-python
        r, g, b = [color.T for color in np.array(im).T]
        print('r', r.shape)
        print('g', g.shape)
        print('b', b.shape)
        row_grid_inds, col_grid_inds, rows_minus_one = GetBox(r)
        box = (col_grid_inds[0], row_grid_inds[0], col_grid_inds[-1], row_grid_inds[-1])
        if show_graphs:
            graph_region_im = im.crop(box)
            graph_region_im.show()
        # graph_region_rim = rim.crop(box)
        # graph_region_rim.show()
        # graph_region_bim = bim.crop(box)
        # graph_region_bim.show()
        # graph_region_gim = gim.crop(box)
        # graph_region_gim.show()
        r_vals = GetGraphVals(r, row_grid_inds, col_grid_inds, True)
        b_vals = GetGraphVals(b, row_grid_inds, col_grid_inds, False)

        # if np.count_nonzero(r_vals == 0):
        #     print(f'r_vals non pos: {r_vals}')
        # if np.count_nonzero(b_vals == 0):
        #     print(f'b_vals non pos: {b_vals}')
        
        grid_height = row_grid_inds[-1] - row_grid_inds[0]
        scaling = rows_minus_one * 200 / grid_height
        r_scaled_vals = r_vals * scaling
        b_scaled_vals = b_vals * scaling
        
        north_v.append(r_scaled_vals)
        south_v.append(b_scaled_vals)

        # print(f'rlen:{len(r_scaled_vals)},blen:{len(b_scaled_vals)},xlen:{len(x_scaled)}')

        # outputname = f"Volumes\\Volumes{name[-14:-4]}.csv"
        # with open(outputname, 'w') as csv_f:
        #     csv_f.write("Date,Hour,Northbound Volume,Southbound Volume")
        #     csv_f.writelines([f'\n1/{int(time // 24 + 1)}/2020,{time%24},{int(round(n_vol))},{int(round(s_vol))}' for time,n_vol,s_vol in zip(x_scaled[:-1],r_scaled_vals,b_scaled_vals)])

        ratio = (col_grid_inds[-1] - col_grid_inds[0]) / grid_height
        if show_graphs:
            fig = plt.figure(figsize=[ratio * 4.8, 4.8])
            ax = fig.gca()
            ax.set_xticks(x_scaled[::4])
            ax.set_yticks(np.linspace(0, rows_minus_one * 200, rows_minus_one + 1))
            # Tight around the box
            ax.set_xlim([x_scaled[0], x_scaled[-1]])
            ax.set_ylim([0, rows_minus_one * 200])
            plt.plot(x_scaled[:-1], r_scaled_vals)
            plt.plot(x_scaled[:-1], b_scaled_vals)
            plt.grid()
            plt.show()
            # plt.savefig(f'Plots\\extracted{name[-14:-4]}.jpg')

    # north_v = [item for sublist in north_v for item in sublist]
    # south_v = [item for sublist in south_v for item in sublist]
    # outputname = f"Volumes\\{signalID}Volumes" + file_base[-8:-1] + ".csv"
    # timestamps = np.arange(0, 24 * month_days, .25)
    # with open(outputname, 'w') as csv_f:
    #     csv_f.write("Date,Hour,Northbound Volume,Southbound Volume")
    #     csv_f.writelines([f'\n1/{int(time // 24 + 1)}/2020,{time%24},{int(round(n_vol))},{int(round(s_vol))}' for time,n_vol,s_vol in zip(timestamps,north_v,south_v)])