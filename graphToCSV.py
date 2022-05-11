"""
Darian Comsa
This is a trial to see if I can extract data points from 
ATSPM volume approach graphs.
Convert the jpeg to rgb arrays and then find the vertex in each column
"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import sys


def GetBooleans(args, file_names):
    output = True
    output_month = False
    show_graphs = False
    save_graphs = False
    overlay = False
    divs = 4
    # Get list of file_names and update boolean.
    if len(args) > 0:
        print(f"\nArguments of the script : {args}")
        if 'show' in args:
            show_graphs = True
            args.remove('show')
        if 'save' in args:
            save_graphs = True
            args.remove('save')
        # 5 minute bins mostly accurate.
        if '5' in args:
            divs = 12
            args.remove('5')
        if 'overlay' in args:
            overlay = True
            args.remove('overlay')
        # To keep reference, use +=.
        file_names += args
    if len(file_names) == 0:
        if not show_graphs:
            output_month = True
        output = False
    return (output, output_month, show_graphs, save_graphs, divs, overlay)


def GetBox(color_band):
    """Find the graph boundaries"""
    row_vals = np.sort(np.sum(color_band, axis=1))
    num_rows = np.count_nonzero(row_vals < 150_000)
    # Each grid row marks 200.
    print(f'num rows: {num_rows}')
    print(row_vals[:num_rows + 1])
    row_grid_inds = np.sort(np.argsort(np.sum(color_band, axis=1))[:num_rows])
    col_grid_inds = np.sort(np.argsort(np.sum(color_band, axis=0))[:25])
    rows_minus_one = num_rows - 1
    grid_height = row_grid_inds[-1] - row_grid_inds[0]
    scaling = rows_minus_one * 200 / grid_height
    return row_grid_inds, col_grid_inds, rows_minus_one, scaling


def GetVal(q, color_band, box_top, box_bot, grid_height, shift_row_inds):
    # Look at column slice width of two pixels, height of grid + 2 extra pixels below.
    pixel_columns = color_band[box_top: box_bot + 3, q: q+2].astype(float)[::-1]
    # White out the grid lines, 2 pixel offset from bottom, edit value as needed [0, 255].
    pixel_columns[grid_height - shift_row_inds + 2] = 190
    # Average between this pixel and the next one up.
    k_av = np.array([np.sum(pixel_columns[pix:pix + 2]/2) for pix in range(grid_height)])#[:-1]
    # A 3x3 grid has its center in middle pixel, 2x2 has center between pixels.
    # Offset by 0.5 - 2.
    return (np.argmin(k_av)-1.5)


def GenerateGraphVals(color_band, row_grid_inds, col_grid_inds, divs):
    """Reads graph points. Yields generator."""
    vals = []
    # Space them properly but the last value is not graphed.
    # box_left = col_grid_inds[0]
    # box_right = col_grid_inds[-1]
    box_top = row_grid_inds[0]
    box_bot = row_grid_inds[-1]
    grid_height = box_bot - box_top
    # Shifted indices start at 0
    shift_row_inds = row_grid_inds - box_top
    # Divide each grid square into divs(=4) columns where to sample data.
    # Slow and exact:
    # quarters = np.concatenate([ np.linspace(c, col_grid_inds[i+1], divs+1)[:-1] for i,c in enumerate(col_grid_inds[:-1]) ]).astype(int)
    # Faster:
    if divs <= 4:
        quarters = np.round(np.concatenate([ np.arange(col_grid_inds[i], c, (c - col_grid_inds[i])/divs)[:divs] for i,c in enumerate(col_grid_inds[1:]) ])).astype(int)
    else:
        # Good enough:
        quarters = np.round(np.concatenate([ np.linspace(col_grid_inds[i], c, divs, endpoint=False) for i,c in enumerate(col_grid_inds[1:]) ])).astype(int)
    # ?:
    # quarters = np.concatenate([ (q-2, q+1) for q in quarters])[1:-1]
    # Do first val, get each side of the middle vals, then last val.
    vals = [GetVal(quarters[0] + 1, color_band, box_top, box_bot, grid_height, shift_row_inds)
        ] + [GetVal(q + k, color_band, box_top, box_bot, grid_height, shift_row_inds) for q in quarters[1:-1] for k in (-2, 1)
        ] + [GetVal(quarters[-1] - 2, color_band, box_top, box_bot, grid_height, shift_row_inds)]
    m = ( (vals[1] - vals[0]) / (quarters[1]-1.5 - (quarters[0]+1.5)) )
    yield max(0, vals[0] - 1.5 * m)
    temp = vals[1] + 1.5 * m
    for i in range(1, len(quarters) - 1):
        # Each quarter is followed by a straight edge, between vals[0] and vals[1], then vals[2] and vals[3], etc.
        # Formulas for line: 
        # y = mx + b
        # m = (y2-y1)/(x2-x1)
        # m = ( (vals[i*2 + 1] - vals[i*2]) / (quarters[i+1]-1.5 - (quarters[i]+1.5)) )
        # y = y1 + m(x - x1)
        # y = vals[i*2] - 1.5*m
        divisor = max((quarters[i+1]-1.5 - (quarters[i]+1.5)), 1)
        m = ( (vals[i*2 + 1] - vals[i*2]) /  divisor )
        yield max(0, (temp + vals[i*2] - 1.5 * m) / 2)
        temp = vals[i*2 + 1] + 1.5 * m
    yield max(0, temp)




if __name__ == "__main__":
    file_names = []
    # Parse command line arguments.
    # output: output .csv, output_month: default no filename, show_graphs: display, save_graphs: save figure, 
    # divisions: hourly divisions, overlay: compare graphs
    output, output_month, show_graphs, save_graphs, divisions, overlay = GetBooleans(sys.argv[1:], file_names)
    # If no filename arguments given, output is false.
    if not output:
        # Default compile month worth of volumes, default January 2020.
        MONTH_DAY_COUNTS = [31,29,31,30,31,30,31,31,30,31,30,31]
        signalID = '6415'
        # Read from graphs in Graphs2020Jan\.
        file_base = "Graphs2020-01\\ApproachVolumesGraph2020-01-"
        month_days = MONTH_DAY_COUNTS[0]
        file_names = [file_base + f"{d+1:02d}.jpg" for d in range(month_days)]
    print(f'output: {output}, showgraphs: {show_graphs}, outputmonth: {output_month}')
    
    # Problem graphs.
    north_v = []
    south_v = []
    x_scaled = np.linspace(0, 24, divisions*24+1)
    for name in file_names[:]:
        print(f'\n{name}')
        im = Image.open(name)
        # rim, gim, bim = im.split()
        # https://stackoverflow.com/questions/10825217/edit-rgb-values-in-a-jpg-with-python
        r, g, b = [color.T for color in np.array(im).T]
        print(f'r {r.shape}')
        print(f'g {g.shape}')
        print(f'b {b.shape}')
        row_grid_inds, col_grid_inds, rows_minus_one, scaling = GetBox(r)
        box = (col_grid_inds[0], row_grid_inds[0], col_grid_inds[-1], row_grid_inds[-1])
        # Shows original graph cropped.
        if show_graphs:
            graph_region_im = im.crop(box)
            graph_region_im.show()

        # Returns numpy array.
        r_vals = GenerateGraphVals(r, row_grid_inds, col_grid_inds, divisions)
        b_vals = GenerateGraphVals(b, row_grid_inds, col_grid_inds, divisions)
        r_vals = tuple(r_vals)
        b_vals = tuple(b_vals)
        r_scaled_vals = np.array(r_vals)*scaling
        b_scaled_vals = np.array(b_vals)*scaling
        
        # Compile month worth of values.
        if output_month:
            north_v.append(r_scaled_vals)
            south_v.append(b_scaled_vals)

        # Output CSV file for one graph.
        if output:
            # outputname = f"TEST\\TEST_Volumes{name[-14:-4]}.csv"
            outputname = f"Volumes\\Volumes{name[-14:-4]}.csv"
            with open(outputname, 'w') as csv_f:
                csv_f.write("Date,Hour,Northbound Volume,Southbound Volume")
                csv_f.writelines([f'\n1/{int(time // 24 + 1)}/2020,{time%24},{int(round(n_vol))},{int(round(s_vol))}' for time,n_vol,s_vol in zip(x_scaled[:-1],r_scaled_vals,b_scaled_vals)])
        
        if show_graphs or save_graphs:
            ratio = (col_grid_inds[-1] - col_grid_inds[0]) / (row_grid_inds[-1] - row_grid_inds[0])
            # 5 inches height is good size.
            fig = plt.figure(figsize=[ratio*5,1*5])
            ax = fig.gca()
            if overlay:
                graph_region_arr = np.array(graph_region_im)
                imx,imy,imd = graph_region_arr.shape
                # print(f'im_array dimensions: {(imx, imy)}')
                extent = (x_scaled[0], x_scaled[-1], 0., rows_minus_one*200)
                plt.imshow(graph_region_arr, extent=extent)
                # plt.imshow(graph_region_arr)
            plt.plot(x_scaled[:-1], tuple(r_scaled_vals), alpha=1)
            plt.plot(x_scaled[:-1], tuple(b_scaled_vals), alpha=1)
            # Add grid lines.
            plt.grid()

            # Do the table of values.
            # https://matplotlib.org/stable/gallery/misc/table_demo.html#sphx-glr-gallery-misc-table-demo-py
            rows = [f':{i:02}' for i in range(0,60,60//divisions)] * 2
            cell_text = [[f'{int(round(r))}' for r in r_scaled_vals[i::divisions]] for i in range(divisions)] + [[f'{int(round(b))}' for b in b_scaled_vals[i::divisions]] for i in range(divisions)]
            colors = [[0,0,1,.2]]*divisions + [[1,0,0,.2]]*divisions
            the_table = plt.table(cellText = cell_text,
                                  rowLabels=rows,
                                  rowColours= colors,
                                #   colLabels=x_scaled[:-1:divisions].astype(int),
                                  bbox=[0, -2/3, 1, 17/32],
                                  loc='bottom')
            plt.subplots_adjust(left=.1, bottom=0.37)

            # Set x and y ticks.
            ax.set_xticks(x_scaled[::divisions])
            ax.set_yticks(np.linspace(0, rows_minus_one * 200, rows_minus_one + 1))
            # Tight around the graphing box.
            ax.set_xlim([x_scaled[0], x_scaled[-1]])
            ax.set_ylim([0, rows_minus_one * 200])
            
            ax.set_aspect(24 / (rows_minus_one * 200 * ratio))

            # Save or show the plot figure.
            if save_graphs:
                # plt.savefig(f'Plots\\TEST_extracted{name[-14:-4]}.jpg')
                plt.savefig(f'Plots\\extracted{name[-14:-4]}.jpg')
            else:
                plt.show()

    # Output CSV file for one month of graphs.
    if output_month:
        # Use generators for memory efficiency.
        north_v_gen = (item for sublist in north_v for item in sublist)
        south_v_gen = (item for sublist in south_v for item in sublist)
        # outputname = f"TEST\\TEST_{signalID}Volumes" + file_base[-8:-1] + ".csv"
        outputname = f"Volumes\\{signalID}Volumes" + file_base[-8:-1] + ".csv"
        timestamps = np.round(np.linspace(0, 24 * month_days, 24 * month_days * divisions), 3)
        with open(outputname, 'w') as csv_f:
            csv_f.write("Date,Hour,Northbound Volume,Southbound Volume")
            csv_f.writelines([f'\n1/{int(time // 24 + 1)}/2020,{time%24},{int(round(n_vol))},{int(round(s_vol))}' for time,n_vol,s_vol in zip(timestamps,north_v_gen,south_v_gen)])
