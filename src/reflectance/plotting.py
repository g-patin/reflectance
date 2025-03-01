import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import colour
from typing import Optional, Union
import imageio
from pathlib import Path
import os

####### DEFINE GENERAL PARAMETERS #######

D65 = colour.CCS_ILLUMINANTS["cie_10_1964"]["D65"]

labels_eq = {
    'dE76': r'$\Delta E^*_{ab}$',
    'dE94': r'$\Delta E^*_{94}$',
    'dE00': r'$\Delta E^*_{00}$',         
    'dR_vis': r'$\Delta R_{vis}$',
    'dL*' : r'$\Delta L^*$',
    'da*' : r'$\Delta a^*$',
    'db*' : r'$\Delta b^*$',
    'dC*' : r'$\Delta C^*$',
    'dh' : r'$\Delta h$',    
    'L*' : r'$L^*$',
    'a*' : r'$a^*$',
    'b*' : r'$b^*$',
    'C*' : r'$C^*$',
    'h' : r'$h$',          
}

x_labels = {
    'Hv': 'Exposure dose $H_v$ (Mlxh)',
    'He': 'Radiant Exposure $H_e$ ($MJ/m^2$)',
    't_s': 'Exposure duration (sec)',
    't_m': 'Exposure duration (min)'
}


ls_dic = {
        'dE76': '--',
        'dE00': '-',
        'dE94': ':',
        'dR_vis': '-.',  
        'L*' : '--',
        'a*' : ':',
        'b*' : '-.', 
        'C*' : '-.', 
        'h' : ':', 
        'dL*' : '-.',
        'da*' : '--',
        'db*' : '-.',
        'dC*' : ':', 
        'dh' : '-', 
        'none' : '-',     
    }


lw_dic = {
        'dE76': 2,
        'dE00': 3,
        'dE94': 1,
        'dR_vis': 2,  
        'L*' : 2,
        'a*' : 2,
        'b*' : 2, 
        'C*' : 1, 
        'h' : 1, 
        'dL*' : 2,
        'da*' : 2,
        'db*' : 2,
        'dC*' : 1, 
        'dh' : 1, 
        'none' : 2,     
    }


colors_dic = {
        'dE76': 'limegreen',
        'dE00': 'blue',
        'dE94': 'yellow',
        'dR_vis': 'green',
        'dL*': 'b',
        'da*': 'red',
        'db*': 'orange', 
        'dC*' : 'brown',
        'dh' : 'grey', 
        'L*' : 'b',
        'a*' : 'red',
        'b*' : 'orange', 
        'C*' : 'grey', 
        'h' : 'brown',        
        'none' : 'k',     
    }


####### THE FUNCTIONS #######

def bars(data, stds=None, coordinate='dE00', colors=None, fontsize=24, legend_labels=[], title=None, title_fontsize=24, save=False, path_fig='cwd'):

    sns.set_theme(font='serif')
    fig, ax = plt.subplots(1,1, figsize=(15,8))

    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    ax.set_xlabel('Microfading analyses numbers', fontsize=fontsize)
    ax.set_ylabel(labels_eq[coordinate], fontsize=fontsize)

    ax.xaxis.grid() # horizontal lines only


    plt.tight_layout()
    plt.show()


def CIELAB(data, stds=None, colors=None, fontsize=24, legend_labels=[], title=None, title_fontsize=24, line=False, legend_position='in', legend_fontsize=20, legend_title='', save=False, path_fig='cwd', start_value=False, dE=False, obs_ill=None, return_data=False, *args, **kwargs):
    """Plot the CIELAB coordinates of one or several datasets.

    Parameters
    ----------
    data : list
        A list of data points, where each data point is a numpy array. 

    stds : list, optional
        A list of standard variation values respective to each element given in the data parameter, by default []

    legend_labels : list, optional
        A list of labels respective to each element given in the data parameter that will be shown in the legend. When the list is empty there is no legend displayed, by default []
    
    title : str, optional
        Add a title, by default None
    
    colors : str, optional
        Color of the data points. When 'sample' is passed as an argument, the color will correspond to the srgb values of the sample. A list of colors - respective to each element given in the data parameter - can be passed, by default None
    
    fontsize : int, optional
        Fontsize of the plot (ticks and labels), by default 24
    
    fontsize_title : int, optional
        Fontsize of the title, by default 24
    
    line : bool, optional
        Add a gray dash line to a time-series of Lab values, by default False
    
    save : bool, optional
        _description_, by default False
    
    path_fig : str, optional
        _description_, by default 'cwd'
    """    

    # fetch the colour circle image
    if dE == False:
        im_colour_circle = imageio.imread(Path(__file__).parent / 'colour_circle.png')

    # set the aesthetics of the figure
    sns.set_theme(font='serif', style='darkgrid', context='paper', palette='colorblind', font_scale=1)

    # create the figure
    figure_sizes = {'in': (10,10), 'out': (12,10)}
    fig, ax = plt.subplots(2,2, figsize=figure_sizes[legend_position], gridspec_kw=dict(width_ratios=[1, 2], height_ratios=[2, 1]))
    Lb, ab, AB, aL = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    
    # define labels 
    if legend_labels == None:
        legend_labels = ['none'] * len(data)        
    elif legend_labels == 'none':        
        legend_labels = ['none'] * len(data)    
    elif len(legend_labels) == 0:
        legend_labels = ['none'] * len(data)
        
    # define std values
    if stds is None:
        stds = [np.zeros(3) for _ in data]   
    
    
    # plot the data
    for i, (el_data, label, std) in enumerate(zip(data, legend_labels, stds)):
        
        # compute dE values
        if dE:            
            dE00 = np.round(np.array([colour.delta_E(el_data[0][:3], d[:3]) for d in el_data]),3)

        # retrieve the Lab values 
        #el_data = el_data.transpose()       
        L, a, b = el_data[0], el_data[1], el_data[2]
                
        # retrieve the light dose values
        H = el_data[3] if dE else None
               
        # define the colors and color_line of the markers        
        if colors == 'sample':
            Lab = np.array([L, a, b]).transpose()               
            srgb = colour.XYZ_to_sRGB(colour.Lab_to_XYZ(Lab), D65).clip(0, 1)
            color = srgb                     
            
        elif colors == None:
            color = None           

        elif isinstance(colors, str):            
            color = colors            

        else:
            color = colors[i]            
        
        
     
        # plot single colour points or grouped colour points
        if len(el_data.shape) == 1:  #.shape
            
            Lb.errorbar(L, b, yerr=std[2], xerr=std[0], fmt='o', color=color, **kwargs)
            ab.errorbar(a, b, yerr=std[2], xerr=std[1], fmt='o', color=color, **kwargs, label=label)
            aL.errorbar(a, L, yerr=std[2], xerr=std[0], fmt='o', color=color, **kwargs) 
                        
            AB.imshow(im_colour_circle, extent=(-110,110,-110,110))  
            AB.scatter(a,b, color='0.5', marker='o') 
            AB.axhline(0, color="black", lw=0.5)
            AB.axvline(0, color="black", lw=0.5)                       

        else:             
            Lb.errorbar(L, b, yerr=std[2], xerr=std[0], fmt='o', color=color, **kwargs)
            ab.scatter(a, b, color=color, label = label, **kwargs)            
            aL.scatter(a, L, color=color, **kwargs)

            

            # plot the dE values or the a*b* values
            if dE:
                AB.plot(H,dE00, color=color)
            else:
                               
                AB.imshow(im_colour_circle, extent=(-110,110,-110,110)) 
                AB.scatter(a,b, color='0.5', marker='o') 
                AB.axhline(0, color="black", lw=0.5)
                AB.axvline(0, color="black", lw=0.5)
                  
            # plot a line connecting colour points
            if line:
                Lb.plot(L,b, color='0.6', ls='--', lw=1)
                ab.plot(a,b, color='0.6', ls='--', lw=1)
                aL.plot(a,L, color='0.6', ls='--', lw=1)


    if dE:
        AB.set_xlim(0)
        AB.set_ylim(0) 

        AB.set_xlabel("Exposure dose $H_v$ (Mlxh)", fontsize=fontsize)
        AB.set_ylabel("$\Delta E^*_{00}$", fontsize=fontsize)  

    else:
        AB.grid(False) 
        AB.set_xlim(-110, 110)
        AB.set_ylim(-110, 110)  

        AB.set_xlabel("CIE $a^*$", fontsize=fontsize)
        AB.set_ylabel("CIE $b^*$", fontsize=fontsize)       
                     
    Lb.set_xlabel("CIE $L^*$", fontsize=fontsize)
    Lb.set_ylabel("CIE $b^*$", fontsize=fontsize)    
    aL.set_xlabel("CIE $a^*$", fontsize=fontsize)
    aL.set_ylabel("CIE $L^*$", fontsize=fontsize) 

    for axis in [Lb, aL, AB, ab]:
        axis.xaxis.set_tick_params(labelsize=fontsize)
        axis.yaxis.set_tick_params(labelsize=fontsize)
 

    if start_value:
        aL.set_title('x : start values', fontsize=title_fontsize)

    if title != None:
        plt.suptitle(title, fontsize=title_fontsize)

    

    if legend_labels[0] != 'none' and len(legend_labels) < 19:
        if legend_position == 'in':
            ab.legend(loc = 'best', fontsize=legend_fontsize, title=legend_title, title_fontsize=legend_fontsize)

        elif legend_position == 'out':            
            ab.legend(loc='upper left',fontsize=legend_fontsize, title=legend_title, bbox_to_anchor=(1, 1), title_fontsize=legend_fontsize)
         

    plt.tight_layout()
    
    if save == True:
        if path_fig == 'cwd':
            path_fig = f'{os.getcwd()}/CIELAB.png'                    
            
        fig.savefig(path_fig,dpi=300, facecolor='white')  
    
    
    if return_data:    
        return plt, Lb,ab,aL
    
    else:
        plt.show()


def delta(data: list, yerr=None, dose_unit:Optional[list] = ['He'], coordinates:Optional[list] = ['dE00'], initial_values=None, object_ids=None, figsize=(15,9), colors=None, ls='random', lw='default', title=None, fontsize=28, legend_labels=[], legend_title=None, legend_fontsize=24, save=False, path_fig='cwd'):
    """Plot the delta values of choosen colorimetric coordinates

    Parameters
    ----------
    data : a list of list
        _description_

    yerr : _type_, optional
        _description_, by default None

    dose_unit : Optional[list], optional
        _description_, by default ['He']

    coordinates : Optional[list], optional
        _description_, by default ['dE00']

    legend_labels : list, optional
        _description_, by default []

    initial_values : _type_, optional
        _description_, by default None

    figsize : tuple, optional
        _description_, by default (15,9)

    colors : _type_, optional
        _description_, by default None

    ls : str, optional
        _description_, by default 'random'

    lw : int, optional
        _description_, by default 2

    title : _type_, optional
        _description_, by default None

    title_legend : _type_, optional
        _description_, by default None

    fontsize : int, optional
        _description_, by default 28

    save : bool, optional
        _description_, by default False

    path_fig : str, optional
        _description_, by default 'cwd'

    Returns
    -------
    _type_
        _description_
    """ 
    
    # define y-std values
    if yerr is None:
        yerr = []
        for ele in data:
            yerr.append([np.zeros(len(x)) for x in ele])

    
    # define the color of the lines
    if colors is None:
        colors = [[colors] * len(coordinates)] * len(data)

    elif colors == 'default':
        colors = [[colors_dic[x] for x in coordinates]] * len(data)

    elif isinstance(colors, str):
        colors = [[colors] * len(coordinates)] * len(data)

    elif type(colors) == list:      
        if len(coordinates) == 1:
            colors = [len(coordinates)*[x] for x in colors] 
        else:
            colors = [colors] * len(data) 


    # define the width of the lines
    if lw == 'default':
        list_lw = [[lw_dic[x] for x in coordinates]] * len(data)  

    elif isinstance(lw, int):
        list_lw = [[lw] * len(coordinates)] * len(data)

    elif isinstance(lw, list):
        if len(coordinates) == 1:
            list_lw = [len(coordinates)*[x] for x in lw] 
        else:
            list_lw = [lw] * len(data)
    
    
    
    # check whether the length of the data matches the length of the dose_unit and coordinates lengths
    for d in data:
        if len(dose_unit + coordinates) != len(d):
            print('The length of each data objects should correspond to sum of the x and y units.')
            return       

        else:
            pass         
    
        
    # set the labels of the legend
    if len(legend_labels) == 0:
        legend_labels = ['none'] * len(data)

    if len(coordinates) == 1:
        legend_labels = [[x] for x in legend_labels]
        list_ls = [[None]] * len(data)
        coordinates = [f'd{x}' if x in ['L*','a*','b*','C*','h'] else x for x in coordinates]
        
    else:
        list_ls = [[ls_dic[x] for x in coordinates]] * len(data)

        if len(initial_values) > 0:            
            
            dy_unit = [f'd{x}' if x in ['L*','a*','b*','C*','h'] else x for x in coordinates]
            dy_unit = [labels_eq[x] for x in dy_unit]
            #y_unit = [f'{x} ({x[1:]} init$ = {i})' for x,i in zip(y_unit, initial_values)]
            dy_unit = [f'{x} (${c[0]}^*_i$ = {i})' for x,c,i in zip(dy_unit,coordinates, initial_values)]

            #dy_unit = [f'{x} (${c[0]}^*_i$ = {i})' if x in ['L*','a*','b*','C*','h'] else x for x in zip(initial_values.keys(), initial_values.values())]
            dy_unit = [f'{labels_eq[x]} (${x[1]}^*_i$ = {np.round(initial_values[x[1:]],1)})' if x in ['dL*','da*','db*','dC*','dh'] else labels_eq[x] for x in coordinates]

        else:
            dy_unit = [f'd{x}' if x in ['L*','a*','b*','C*','h'] else x for x in coordinates]        
            dy_unit = [labels_eq[x] for x in dy_unit]

        legend_labels = [dy_unit] * len(data)
        

    
    # Set the aesthetics of the figure
    sns.set_theme(context='paper', font='serif', palette='colorblind') 

    # Plot with a single x-axis, ie. 1 light energy unit
    if len(dose_unit) == 1:

        # create an empty figure
        fig, ax1 = plt.subplots(1,1, figsize=figsize)
        
        # define the random linestyles
        if ls == 'random':
            plt.rcParams['axes.prop_cycle'] = ("cycler('ls', ['-', '--', ':', '-.'])")
            ax1.set_prop_cycle(ls = ["-","--","-.",":"])

        
        for d,s,label,ls,lw,color in zip(data,yerr,legend_labels,list_ls,list_lw,colors):

            x = d[0]            

            for y,s_val,l,ls_val,lw_val,c in zip(d[1:],s,label,ls,lw,color):
               
                if ls == 'random':
                    ax1.plot(x, y, lw=lw_val, color=c, label=l)
                else:
                    ax1.plot(x, y, lw=lw_val, ls=ls_val, color=c, label=l)
                
                ax1.fill_between(x, y+s_val, y-s_val, alpha=0.5, color='0.75', ec='none')

        handles, list_labels = ax1.get_legend_handles_labels()        
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, list_labels)) if l not in list_labels[:i]]        
            
        ax1.legend(*zip(*unique), fontsize=legend_fontsize, title=legend_title, title_fontsize=legend_fontsize)

        ax1.set_xlim(0)

        ax1.xaxis.set_tick_params(labelsize=fontsize)
        ax1.yaxis.set_tick_params(labelsize=fontsize)

        ax1.set_xlabel(x_labels[dose_unit[0]], fontsize=fontsize)

        if len(coordinates) == 1:
            ax1.set_ylabel(labels_eq[coordinates[0]], fontsize=fontsize)

        else:
            ax1.set_ylabel('Colorimetric differences ($\Delta$)', fontsize=fontsize)

        
            

    # Plot with two x-axes, ie. 2 light energy units
    else:

        print('double x axis')
        ax2 = ax1.twiny()


        '''
        if legend == 'coordinates':
            
            list_objects = sorted(set(data.columns.get_level_values(0)))        

            for object in list_objects:
                d = data[object].dropna()            
                x = d.iloc[:,0]

                
                for col in d.iloc[:,1:].columns:
                    y = d[col]
                    
                    ax.plot(x,y, color=colors_dic[col], label=col)

            handles, labels = ax.get_legend_handles_labels()
            labels = [labels_eq[x] for x in labels]
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            
            ax.legend(*zip(*unique), fontsize=fontsize)

        
        elif legend == 'objects':

            list_objects = sorted(set(data.columns.get_level_values(0)))
            


            for object in list_objects:
                d = data[object].dropna()
                coordinate = d.columns[1]

                x = d.iloc[:,0]
                y = d.iloc[:,1]

                ax.plot(x,y, label=object)

                ax.legend(fontsize=fontsize)        

        
    
    
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)

        #ax.set_xlabel(x_labels[x_scale[0]], fontsize=fontsize)

        ax.set_xlim(0)

        
        #plt.legend()
        plt.tight_layout()
        plt.show()
        '''

    ax1.set_title(title, fontsize=fontsize)
    plt.tight_layout()
    
    if save == True:
        if path_fig == 'cwd':
            path_fig = f'{os.getcwd()}/dE.png'                    
            
        fig.savefig(path_fig,dpi=300, facecolor='white')         

    if len(dose_unit) == 1: 
        return plt, ax1
    else:
        return plt, ax1, ax2


def spectra(data, stds=[], spectral_mode:Optional[str] = 'R', legend_labels=[], title='none', fontsize=24, fontsize_legend:Optional[int] = 22, legend_title='', x_range=(), colors:Union[str, list] = None, lw:Optional[int] = 2, ls:Union[str, list] = '-', text:Optional[str] = '', save=False, path_fig='cwd', derivation=False, *args, **kwargs):
    """
    Description: Plot the reflectance spectrum of one or several datasets.

    
    Args:
        _ data (list): A list of data elements, where each element corresponding to a reflectance spectrum is a numpy array. 

        _ std (list, optional): A list of standard variation values respective to each element given in the data parameter. Defaults to [].

        spectral_mode : string, optional
            When 'R', it diplays the y-axis label for reflectance spectra
            When 'dR', it displays the y-axis label for the difference in reflectance values
            When 'A', it displays the y-axis label for absorption spectra using the following equation: A = -log(R)

        _ labels (list, optional): A list of labels respective to each element given in the data parameter that will be shown in the legend. When the list is empty there is no legend displayed. Defaults to [].
        
        _ title (str, optional): Suptitle of the figure. When 'none' is passed as an argument, there is no suptitle displayed. Defaults to 'none'.
        
        _ color_data (str or list, optional): Color of the data points. When 'sample' is passed as an argument, the color will correspond to the srgb values of the sample. A list of colors - respective to each element given in the data parameter - can be passed. Defaults to 'sample'.
        
        _ fs (int, optional): Fontsize of the plot (title, ticks, and labels). Defaults to 24.    

        _ x_range (tuple, optional): Lower and upper limits of the x-axis. Defaults to (). 

                    
    Returns: A figure showing the reflectance spectra.
    """    
    
    # Set the observer and illuminant
    observer = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER["CIE 1964 10 Degree Standard Observer"] 
    illuminant = colour.SDS_ILLUMINANTS['D65'] 
    d65 = colour.CCS_ILLUMINANTS["cie_10_1964"]["D65"]
    
    # Create the figure
    sns.set_theme(context='paper', font='serif', palette='colorblind')
    fig, ax = plt.subplots(1,1, figsize=(15, 9))
    
    # Set the list of labels
    if len(legend_labels) == 0:
        legend_labels = ['none'] * len(data)

    
    # Set the list of colors
    if isinstance(colors, list) or isinstance(colors, np.ndarray):        
        colors = colors        

    elif colors == None:
        colors = [None] * len(data)
    
    elif colors == 'sample':
        colors = ['sample'] * len(data)
    
    # Set the linestyle
    if isinstance(ls, str):
        ls = [ls] * len(data)

    # Set the linewidth
    if isinstance(lw, int):        
        lw = [lw] * len(data)
    
    # Set the std values
    if len(stds) == 0:        
        stds = [np.zeros(len(x[1])) for x in data]
                
         
    # Initiate a for loop to plot the data    
    for i, (d,s) in enumerate(zip(data,stds)):
        
        df_sp = pd.DataFrame(data=[d[1],s], columns=d[0], index=['sp','std']).T

        # Index data according the x_range values
        if x_range not in [(), None]:            
            df_sp = df_sp.loc[x_range[0]:x_range[1]]

        # Get the wavelengths and spectral values
        wl = df_sp.index.values
        sp = df_sp.iloc[:,0].values
        std = df_sp['std'].values        

        if isinstance(colors, list) or isinstance(colors, np.ndarray):
            color = colors[i]
        
        elif colors[i] == 'sample':                      
            sd = colour.SpectralDistribution(sp,wl)  
            XYZ = colour.sd_to_XYZ(sd,observer, illuminant=illuminant) 
            srgb = colour.XYZ_to_sRGB(XYZ / 100, illuminant=d65).clip(0, 1)
            color = np.array(srgb)            
        
               
                
        ax.plot(wl,sp, color=color, lw=lw[i], ls=ls[i], label=legend_labels[i])
        ax.fill_between(wl, sp-std,sp+std, alpha=0.5, color='0.75', ec='none')
        
        
    if x_range not in [(), None]:
        ax.set_xlim(x_range[0],x_range[1])
    
    ax.set_xlabel('Wavelength $\lambda$ (nm)', fontsize = fontsize)

    if derivation == False and spectral_mode.lower() == 'r':
        ax.set_ylabel('Reflectance factor', fontsize = fontsize)
    elif derivation == False and spectral_mode.lower() == 'dr':
        ax.set_ylabel('Reflectance difference', fontsize = fontsize)
    elif derivation == False and spectral_mode.lower() == 'a':
        ax.set_ylabel('Absorbance', fontsize = fontsize)
    elif derivation == True and spectral_mode.lower() == 'a':
        ax.set_ylabel(r'$\frac{dA}{d\lambda}$', fontsize = fontsize+10)
    else:
        ax.set_ylabel(r'$\frac{dR}{d\lambda}$', fontsize = fontsize+10)

    ax.xaxis.set_tick_params(labelsize = fontsize)
    ax.yaxis.set_tick_params(labelsize = fontsize)

    if title != 'none':
        ax.set_title(title, fontsize = fontsize+3)
    
    if len(legend_labels) > 6:
        ncols = 2
    else:
        ncols = 1

    if legend_labels[0] != 'none' and len(legend_labels) < 19:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  
        #plt.legend(labels, fontsize=fontsize_legend, title='Measurement $n^o$', title_fontsize=fontsize_legend) 
        plt.legend(by_label.values(), by_label.keys(), ncol=ncols, fontsize=fontsize_legend, title=legend_title, title_fontsize=fontsize_legend)

    
    if text != '':
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.01,0.05,text,transform=ax.transAxes,fontsize=fontsize-6,verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save == True:
        if path_fig == 'cwd':
            path_fig = f'{os.getcwd()}/SP.png'                    
            
        fig.savefig(path_fig,dpi=300, facecolor='white')       

    plt.show()


def swatches_circle(data, data_type:Optional[str] = 'Lab', light_doses: Optional[list] = [0.5,1,2,5,15], JND:Optional[list] = [], dE:Optional[bool] = True, fontsize: Optional[int] = 24, save:Optional[bool] = False, path_fig:Optional[str] = 'cwd', title:Optional[str] = None, background_grey: Optional [float] = 0.85):

    if list(set([len(x) for x in data]))[0] == len(JND):
        xlabel = 'Just noticeable difference (JND)'

    elif list(set([len(x) for x in data]))[0] == len(light_doses):
        xlabel = 'Exposure dose $H_v$ (Mlxh)'

    else:
        print('Plotting aborted ! The length of data values is not equal to the length of light_doses or JND values.')
        return
    
    if data_type.lower() == 'lab':          
        data_srgb = [colour.XYZ_to_sRGB(colour.Lab_to_XYZ(x), D65).clip(0, 1) for x in data]
    else:
        data_srgb = data
    
    if dE:
        if data_type.lower() == 'lab':
            dE_values = [np.round(colour.delta_E(x[0],x[1:]),1) for x in data]                      

        else:
            print('Plot aborted. Please provide the Lab values instead.')
            return None
    else:
        dE_values = [[''] * (len(light_doses)-1)] * len(data_srgb)

    if isinstance(title, list):
        title = title
    
    elif title == None:
        title = [''] * len(data_srgb)
    
    elif isinstance(title, str):
        title = [title] * len(data_srgb)   
    
    nb = 1

    for d_srgb,dE_val, title_value in zip(data_srgb, dE_values, title):

        N = len(d_srgb)
        fig, ax = plt.subplots(1,1, figsize=((N-1)*5,6))        

        ax.set_facecolor((background_grey,background_grey,background_grey))
        fig.patch.set_facecolor((background_grey, background_grey, background_grey))

        if isinstance(title, str):
            title_space = 0.05
        else:
            title_space = 0

        if dE:
            y = 1
            h = 0.7 + title_space
        else:
            y =0.9
            h = 0.6 + title_space

        cp_init = matplotlib.patches.Rectangle((0.05, 0.0), 0.9, y, edgecolor='None', fc=d_srgb[0], lw=2)
        ax.add_patch(cp_init)

        i = 0
        for d in d_srgb[1:]:
            cp = matplotlib.patches.Ellipse(xy=(1/N + i, 0.5), width=0.6/(N-1), height=h, edgecolor='None', fc=d, lw=2)
            ax.add_patch(cp)
            i = i + (1/N)

        ax.xaxis.set_ticks_position(position='bottom')
        ax.set_xticks(np.linspace(0,1,N+1)[1:-1])        
        ax.set_xticklabels(light_doses[1:])
        ax.set_yticks([])

        ax.xaxis.set_tick_params(labelsize=fontsize)    
        ax.set_xlabel('Exposure dose $H_v$ (Mlxh)', fontsize=fontsize)

        ax.set_title(title_value, fontsize=fontsize+2)

        if dE:
            ax_top = ax.secondary_xaxis('top')
            ax_top.set_xlabel('$\Delta E^*_{00}$ values', fontsize=fontsize)
            ax_top.set_xticks(np.linspace(0,1,N+1)[1:-1]) 
            ax_top.set_xticklabels(dE_val)
            ax_top.xaxis.set_tick_params(labelsize=fontsize) 
            ax_top.spines['top'].set_visible(False) 
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.grid(False)       

        plt.tight_layout()
            
        if save == True:
            if path_fig == 'cwd':
                path_fig = f'{os.getcwd()}/MFT_{str(nb).zfill(2)}_SW.png'  
                nb = nb + 1  
                             
                
            fig.savefig(path_fig,dpi=300, facecolor=(background_grey, background_grey, background_grey)) 
        
        plt.show()