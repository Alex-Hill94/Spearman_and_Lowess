import matplotlib as mpl
mpl.use('agg')
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from moving_spearman import spearman as SP
from matplotlib import rc
matplotlib.colorbar.ColorbarBase
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from paul_tol.paul_tol_colours import *
from scipy.stats import binned_statistic_2d
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

class Plot_SP_Rank():
    def __init__(

        self,
        xs= [],
        ys = [],
        zs = []):
        self.xs = xs
        self.ys = ys
        self.zs = zs


    def compute_sp(self, window_sizes=[300,100],
                            window_steps=[50,25],
                            transition_points=[12.5,]):
        ###Computes the Spearman Rank coefficient for given x, y and z values. Futher computes
        ###the residuals of the z value to colour data points by
 
        x, y, z = self.xs, self.ys, self.zs
        sorts = np.argsort(x)
        self.X, self.Y, self.Z = x[sorts], y[sorts], z[sorts] # Ordered x, y and z values

        L_z = lowess(endog = self.zs, exog = self.xs, return_sorted = True)
        self.L_z = L_z
        self.z_residual_plot = self.Z - L_z[:,1]

        L_y = lowess(endog = self.ys, exog = self.xs, return_sorted = True)
        y_residual = self.Y - L_y[:,1]


        centres, moving_rank, moving_pvalue = SP.get_moving_spearman_rank(self.X, y_residual, self.z_residual_plot, window_sizes=window_sizes, window_steps=window_steps, transition_points=transition_points)

        self.centres, self.moving_rank, self.moving_pvalue = centres, moving_rank, moving_pvalue

        self.X_LOCS = self.X[self.centres]

    def plot_me(self, file_name = 'figure', x_lab = 'Dummy', y_lab = 'Dummy', z_lab = 'Dummy', save = False, FS = 20):

        grid = (6,7)
        big_col = grid[0]
        big_row = grid[0] - 1
        small_col = big_col
        small_row = grid[0]-big_row
        loc_big_1 = (0, 0)
        loc_hist_1 = (big_row, 0)    
        significance = 0.05



        ## Generates the colour map for the x vs y point, coloured by the residual z value 
        z_col = self.z_residual_plot
        #z_col = self.Z
        cmap=tol_cmap('rainbow_PuRd')#matplotlib.cm.get_cmap('RdYlGn')
        ### vmin vmax assumes mean_z_col ~ 0
        vmin, vmax = -1.*np.std(z_col), 1.*np.std(z_col)
        #vmin, vmax = np.min(z_col), np.max(z_col)#-2.*np.nanstd(z_col), 2.*np.nanstd(z_col)
        norm=matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        rgba=cmap(norm(z_col))
        XLIM = [min(self.X), max(self.X)]
        #XLIM = [11.95, 12.05]
        diff = abs(min(self.Y) - max(self.Y))
        #YLIM = [min(self.Y) - diff/100., max(self.Y)]
        #YLIM = [0.73, 2.1]
        ## Median line for x vs y relation
        L_y = lowess(endog = self.ys, exog = self.xs)#, frac=0.2, it=3, delta=0.0)
        #lowess.lowess(ys, xs, frac=0.2, it=3, delta=0.0, is_sorted=True, missing='none', return_sorted=False)


        matplotlib.rcParams.update({'font.size': FS})
        fig, ax = plt.subplots(1,1, figsize = [grid[0]+2,grid[1]+1])
        gridspec.GridSpec(grid[0], grid[1])
        box1 = plt.subplot2grid(grid, loc_big_1, colspan = big_col, rowspan = big_row)
        hist1 = plt.subplot2grid(grid, loc_hist_1, colspan = small_col, rowspan = small_row)
       
        # Plots main scatter
        box1.scatter(self.X, self.Y, marker = 'o', s = 5, color = rgba, alpha = 1, rasterized = True)
        # Plots x vs y lowess median line
        box1.plot(L_y[:,0], L_y[:, 1], 'white', linewidth = 2)
        box1.plot(L_y[:,0], L_y[:, 1], 'k', linewidth = 1.5)
        self.L_y = L_y
        # Plots rho
        hist1.plot(XLIM, np.zeros(len(XLIM)), 'k', alpha = 0.6)
        hist1.plot(self.X_LOCS, self.moving_rank)
        hist1.fill_between(self.X_LOCS, -1, 1, where=self.moving_pvalue>significance, facecolor='gray', alpha=0.5)


        # Figure properties
        hist1.set_ylabel('$\\rho$')
        box1.set_xlim(XLIM)
        hist1.set_xlim(XLIM)
        box1.set_ylim(YLIM)
        box1.set_xticklabels([])
        box1.set_ylabel('%s' % y_lab)
        hist1.set_ylim([-1.1, 1.1])
        hist1.set_yticks([-1, 0, 1])
        hist1.set_xlabel('%s' % x_lab)    
        box1.tick_params(which = 'both', direction= 'in', right =True, top = True)
        hist1.tick_params(which = 'both', direction= 'in', right =True, top = True)

        colorbar_ax=fig.add_axes([0.787,0.11,0.02,0.77])
        #left, bottom, width, height
        cb=matplotlib.colorbar.ColorbarBase(colorbar_ax,norm=norm,cmap=cmap,orientation='vertical', extend = 'both')
        cb.set_label('%s' % z_lab,fontsize=16)
        colorVal=cb.to_rgba(self.Z)

        plt.subplots_adjust(hspace = 0)
        if save:
            plt.savefig('%s' % file_name, dpi = 200)
        else:
            plt.show()
        plt.close()

    def plot_hist2d(self, file_name = 'figure', x_lab = 'Dummy', y_lab = 'Dummy', z_lab = 'Dummy', save = False):

        grid = (6,7)
        big_col = grid[0]
        big_row = grid[0] - 1
        small_col = big_col
        small_row = grid[0]-big_row
        loc_big_1 = (0, 0)
        loc_hist_1 = (big_row, 0)    
        significance = 0.05



        ## Generates the colour map for the x vs y point, coloured by the residual z value 
        z_col = self.Z 
        #z_col = self.Z
        cmap=tol_cmap('rainbow_PuRd')#matplotlib.cm.get_cmap('RdYlGn')
        ### vmin vmax assumes mean_z_col ~ 0
        #vmin, vmax = np.min(z_col), np.max(z_col)#-2.*np.nanstd(z_col), 2.*np.nanstd(z_col)
        FS = 20
        XLIM = [min(self.X), max(self.X)]
        #XLIM = [11.95, 12.05]
        diff = abs(min(self.Y) - max(self.Y))
        #YLIM = [min(self.Y) - diff/100., max(self.Y)]
        YLIM = [-0.05, 1.05]
        ## Median line for x vs y relation
        L_y = lowess(endog = self.ys, exog = self.xs)#, frac=0.2, it=3, delta=0.0)
        #lowess.lowess(ys, xs, frac=0.2, it=3, delta=0.0, is_sorted=True, missing='none', return_sorted=False)

        def get_hist2d(points_A, points_B, bins):
            data, bins1, bins2 = np.histogram2d(points_A, points_B, bins = bins, normed = False)
            sum_data = np.sum(data)
            data_norm = data/sum_data
            DAT = np.log10(data_norm).T
            return DAT, bins1, bins2

        def med_hist2d(points_A, points_B, points_C, bins):
            bins1 = np.linspace(min(points_A), max(points_A), bins)
            bins2 = np.linspace(min(points_B), max(points_B), bins)
            DAT, bins1, bins2, bnum = binned_statistic_2d(points_A, points_B, points_C, statistic=np.nanmedian, bins=[bins1, bins2])
            DAT = DAT.T
            return DAT, bins1, bins2

        data, binsa1, binsa2 = get_hist2d(self.X, self.Y, bins = 50)
        data, binsa1, binsa2 = med_hist2d(self.X, self.Y, self.Z, bins = 50)

        def IQR(vals):
            vals_red = vals[np.isnan(vals) == False]
            LQ, UQ   = np.percentile(vals_red, [25, 75])
            IQR = UQ - LQ 
            return LQ, UQ

        vmin, vmax = IQR(self.Z)
        CMAP = tol_cmap('rainbow_PuRd')
        norm=matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        matplotlib.rcParams.update({'font.size': FS})
        fig, ax = plt.subplots(1,1, figsize = [grid[0]+2,grid[1]+1])
        gridspec.GridSpec(grid[0], grid[1])
        box1 = plt.subplot2grid(grid, loc_big_1, colspan = big_col, rowspan = big_row)
        hist1 = plt.subplot2grid(grid, loc_hist_1, colspan = small_col, rowspan = small_row)
        
        #im1 = box1.imshow(data, aspect = 'auto', origin = 'lower', cmap = CMAP, extent = [binsa1[0], binsa1[-1], binsa2[0], binsa2[-1]])
        im1 = box1.hexbin(x = self.X, y = self.Y, C = self.Z, reduce_C_function = np.nanmedian, gridsize = 50, cmap = CMAP)
        box1.plot(L_y[:,0], L_y[:, 1], 'white', linewidth = 2)
        box1.plot(L_y[:,0], L_y[:, 1], 'k', linewidth = 1.5)
        #axs[0].set_ylabel('$\\mathrm{log_{10}}M_{200}\ [M_{\odot}]$')
        #axs[1].scatter(log_r4, self.masses, s = 0.1, marker = 'x', color = 'k', alpha = 0.5)
        #cb1 = plt.colorbar(im1, ax = box1[:], label = '$\\mathrm{log_{10}\ fraction\ of\ objects\ per\ pixel}$', orientation = 'vertical')#, shrink = 0.8)
        #plt.colorbar()
        #cb1 = matplotlib.colorbar(im1, ax = box1, label = '$\\mathrm{log_{10}\ fraction\ of\ objects\ per\ pixel}$', orientation = 'vertical')#, shrink = 0.8)

        colorbar_ax=fig.add_axes([0.787,0.11,0.02,0.77])
        #left, bottom, width, height
        cb=matplotlib.colorbar.ColorbarBase(colorbar_ax,norm=norm,cmap=CMAP,orientation='vertical', extend = 'both')
        cb.set_label('%s' % z_lab,fontsize=16)
        #colorVal=cb.to_rgba(self.Z)

        # Plots main scatter
        # Plots x vs y lowess median line
        hist1.plot(XLIM, np.zeros(len(XLIM)), 'k', alpha = 0.6)
        hist1.plot(self.X_LOCS, self.moving_rank)
        hist1.fill_between(self.X_LOCS, -1, 1, where=self.moving_pvalue>significance, facecolor='gray', alpha=0.5)


        # Figure properties
        hist1.set_ylabel('$\\rho$')
        box1.set_xlim(XLIM)
        hist1.set_xlim(XLIM)
        box1.set_ylim(YLIM)
        box1.set_xticklabels([])
        box1.set_ylabel('%s' % y_lab)
        hist1.set_ylim([-1.1, 1.1])
        hist1.set_yticks([-1, 0, 1])
        hist1.set_xlabel('%s' % x_lab)    
        box1.tick_params(which = 'both', direction= 'in', right =True, top = True)
        hist1.tick_params(which = 'both', direction= 'in', right =True, top = True)

        #colorbar_ax=fig.add_axes([0.787,0.11,0.02,0.77])
        #left, bottom, width, height
        #cb=matplotlib.colorbar.ColorbarBase(colorbar_ax,norm=norm,cmap=cmap,orientation='vertical', extend = 'both')
        #cb.set_label('%s' % z_lab,fontsize=16)
        #colorVal=cb.to_rgba(self.Z)

        plt.subplots_adjust(hspace = 0)
        if save:
            plt.savefig('%s' % file_name, dpi = 200)
        else:
            plt.show()
        plt.close()

