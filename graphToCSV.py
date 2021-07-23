"""
Darian Comsa
This is a trial to see if I can extract data points from 
ATSPM volume approach graphs.
Convert the jpeg to rgb arrays and then find the vertex in that column
"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import sys


def Booleans(args, file_names):
    output = True
    output_month = False
    show_graphs = False
    save_graphs = False
    # Get list of file_names and update boolean
    if len(args) > 0:
        print(f"\nArguments of the script : {args}")
        if 'show' in args:
            show_graphs = True
            args.remove('show')
        if 'save' in args:
            save_graphs = True
            args.remove('save')
        # To keep reference, use +=
        file_names += args
    if len(file_names) == 0:
        if not show_graphs:
            output_month = True
        output = False
    return (output, output_month, show_graphs, save_graphs)


def GetBox(color_band):
    """Find the graph boundaries"""
    row_vals = np.sort(np.sum(color_band, axis=1))
    num_rows = np.count_nonzero(row_vals < 150_000)
    # Each grid row marks 200
    print(f'num rows: {num_rows}')
    print(row_vals[:num_rows + 1])
    row_grid_inds = np.sort(np.argsort(np.sum(color_band, axis=1))[:num_rows])
    col_grid_inds = np.sort(np.argsort(np.sum(color_band, axis=0))[:25])
    rows_minus_one = num_rows - 1
    grid_height = row_grid_inds[-1] - row_grid_inds[0]
    scaling = rows_minus_one * 200 / grid_height
    return row_grid_inds, col_grid_inds, rows_minus_one, scaling


def GetVal(q, color_band, box_top, box_bot, grid_height, shift_row_inds, scaling):
    # Look at column slice width of two pixels, height of grid + 2 extra pixels
    pixel_columns = color_band[box_top: box_bot + 3, q: q+2].astype(float)[::-1]
    # White out the grid lines, 2 pixel offset from bottom, edit value 190 as needed
    pixel_columns[grid_height - shift_row_inds + 2] = 190
    # Average between this pixel and the next one up
    k_av = np.array([np.sum(pixel_columns[pix:pix + 2]/2) for pix in range(grid_height)])#[:-1]
    # A 3x3 grid has its center in middle pixel, 2x2 has center between pixels
    # Offset by 0.5 - 2
    return (np.argmin(k_av)-1.5)*scaling


def GenGraphVals(color_band, row_grid_inds, col_grid_inds, scaling, is_red):
    """Reads graph points"""
    vals = []
    # Space them properly but the last value is not graphed
    # box_left = col_grid_inds[0]
    # box_right = col_grid_inds[-1]
    box_top = row_grid_inds[0]
    box_bot = row_grid_inds[-1]
    grid_height = box_bot - box_top
    # Shifted indices start at 0
    shift_row_inds = row_grid_inds - box_top
    # Divide each grid square into 4 columns where to sample data
    # Slow and exact
    # quarters = np.concatenate([ np.linspace(c, col_grid_inds[i+1], 5)[:-1] for i,c in enumerate(col_grid_inds[:-1]) ]).astype(int)
    # Fast and good enough
    quarters = np.round(np.concatenate([ np.arange(col_grid_inds[i], c, (c - col_grid_inds[i])/4)[:4] for i,c in enumerate(col_grid_inds[1:]) ])).astype(int)
    # quarters = np.concatenate([ (q-2, q+1) for q in quarters])[1:-1]
    # Do first val, middle vals, then last val
    vals = [GetVal(quarters[0] + 1, color_band, box_top, box_bot, grid_height, shift_row_inds, scaling)
    ] + [GetVal(q + k, color_band, box_top, box_bot, grid_height, shift_row_inds, scaling) for q in quarters[1:-1] for k in (-2, 1)
    ] + [GetVal(quarters[-1] - 2, color_band, box_top, box_bot, grid_height, shift_row_inds, scaling)]
    newvals = []
    m = ( (vals[1] - vals[0]) / (quarters[1]-1.5 - (quarters[0]+1.5)) )
    yield max(0, vals[0] - 1.5 * m)
    temp = vals[1] + 1.5 * m
    for i in range(1, len(quarters) - 1):
        # Each quarter is followed by a straight edge, between vals[0] and vals[1], then vals[2] and vals[3], etc
        # Formulas for line: 
        # y = mx + b
        # m = (y2-y1)/(x2-x1)
        # m = ( (vals[i*2 + 1] - vals[i*2]) / (quarters[i+1]-1.5 - (quarters[i]+1.5)) )
        # y = y1 + m(x - x1)
        # y = vals[i*2] - 1.5*m
        m = ( (vals[i*2 + 1] - vals[i*2]) / (quarters[i+1]-1.5 - (quarters[i]+1.5)) )
        yield max(0, (temp + vals[i*2] - 1.5 * m) / 2)
        temp = vals[i*2 + 1] + 1.5 * m
    yield max(0, temp)




if __name__ == "__main__":
    file_names = []
    output, output_month, show_graphs, save_graphs = Booleans(sys.argv[1:], file_names)
    if not output:
        # Default compile month worth of volumes, default January 2020
        MONTH_DAY_COUNTS = [31,29,31,30,31,30,31,31,30,31,30,31]
        signalID = '6415'
        # Read from graphs in Graphs2020Jan
        file_base = "Graphs2020Jan\\VolumesGraph2020-01-"
        month_days = MONTH_DAY_COUNTS[0]
        file_names = [file_base + f"{d+1:02d}.jpg" for d in range(month_days)]
    print(f'output: {output}, showgraphs: {show_graphs}, outputmonth: {output_month}')
    
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
        print(f'r {r.shape}')
        print(f'g {g.shape}')
        print(f'b {b.shape}')
        row_grid_inds, col_grid_inds, rows_minus_one, scaling = GetBox(r)
        box = (col_grid_inds[0], row_grid_inds[0], col_grid_inds[-1], row_grid_inds[-1])
        # Shows original graph cropped
        if show_graphs:
            graph_region_im = im.crop(box)
            graph_region_im.show()

        # Returns numpy array
        r_scaled_vals = GenGraphVals(r, row_grid_inds, col_grid_inds, scaling, is_red=True)
        b_scaled_vals = GenGraphVals(b, row_grid_inds, col_grid_inds, scaling, is_red=False)
        r_scaled_vals = tuple(r_scaled_vals)
        b_scaled_vals = tuple(b_scaled_vals)
        
        # Compile month worth of values
        if output_month:
            north_v.append(r_scaled_vals)
            south_v.append(b_scaled_vals)

        # Output CSV file for one graph
        if output:
            # outputname = f"TEST\\TEST_Volumes{name[-14:-4]}.csv"
            outputname = f"Volumes\\Volumes{name[-14:-4]}.csv"
            with open(outputname, 'w') as csv_f:
                csv_f.write("Date,Hour,Northbound Volume,Southbound Volume")
                csv_f.writelines([f'\n1/{int(time // 24 + 1)}/2020,{time%24},{int(round(n_vol))},{int(round(s_vol))}' for time,n_vol,s_vol in zip(x_scaled[:-1],r_scaled_vals,b_scaled_vals)])
        
        if show_graphs or save_graphs:
            ratio = (col_grid_inds[-1] - col_grid_inds[0]) / (row_grid_inds[-1] - row_grid_inds[0])
            # 5 inches height is good size
            fig = plt.figure(figsize=[ratio * 5, 25/3])
            ax = fig.gca()
            plt.plot(x_scaled[:-1], tuple(r_scaled_vals))
            plt.plot(x_scaled[:-1], tuple(b_scaled_vals))
            plt.grid()
            # https://matplotlib.org/stable/gallery/misc/table_demo.html#sphx-glr-gallery-misc-table-demo-py
            rows = [f':{i:02}' for i in range(0,60,15)] * 2
            cell_text = [[f'{int(round(r))}' for r in r_scaled_vals[i::4]] for i in range(4)] + [[f'{int(round(b))}' for b in b_scaled_vals[i::4]] for i in range(4)]
            colors = [[0,0,1,.2]]*4 + [[1,0,0,.2]]*4
            the_table = plt.table(cellText = cell_text,
                                  rowLabels=rows,
                                  rowColours= colors,
                                #   colLabels=x_scaled[:-1:4].astype(int),
                                  bbox=[0, -2/3, 1, 16/27],
                                  loc='bottom')
            plt.subplots_adjust(left=0.1, bottom=0.4)
            # Set x and y
            ax.set_xticks(x_scaled[::4])
            ax.set_yticks(np.linspace(0, rows_minus_one * 200, rows_minus_one + 1))
            # Tight around the graphing box
            ax.set_xlim([x_scaled[0], x_scaled[-1]])
            ax.set_ylim([0, rows_minus_one * 200])
            # Save or show the plot figure
            if save_graphs:
                # plt.savefig(f'Plots\\TEST_extracted{name[-14:-4]}.jpg')
                plt.savefig(f'Plots\\extracted{name[-14:-4]}.jpg')
            else:
                plt.show()

    # Output CSV file for one month of graphs
    if output_month:
        # Use generators for memory efficiency
        north_v_gen = (item for sublist in north_v for item in sublist)
        south_v_gen = (item for sublist in south_v for item in sublist)
        # outputname = f"TEST\\TEST_{signalID}Volumes" + file_base[-8:-1] + ".csv"
        outputname = f"Volumes\\{signalID}Volumes" + file_base[-8:-1] + ".csv"
        timestamps = np.arange(0, 24 * month_days, .25)
        with open(outputname, 'w') as csv_f:
            csv_f.write("Date,Hour,Northbound Volume,Southbound Volume")
            csv_f.writelines([f'\n1/{int(time // 24 + 1)}/2020,{time%24},{int(round(n_vol))},{int(round(s_vol))}' for time,n_vol,s_vol in zip(timestamps,north_v_gen,south_v_gen)])
