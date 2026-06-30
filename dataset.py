import pandas as pd
import numpy as np
import h5py

wl = np.arange(380, 781, 1)



data_folder = "data/"
# data_folder = "/media/nchudnov/extra_sp/stu/code/data/"
illuminant_folder = "illuminants/"
reflectances_folder = "reflectances/"
spectral_sensitivity_folder = "spectral_sensitivity/"

class Dataset(object):
    def __init__(self):
        self.illuminant = self.illuminant()
        self.reflectances = self.reflectances()
        self.spectral_sensitivity = self.spectral_sensitivity()

    def load_hyperspectral_image(self, path):
        f = h5py.File(path, 'r')
        wl = np.array(f['wavelengths'])
        radiance = np.array(f['radiance']).astype(np.float64)
        illum = np.array(f['illuminant']).astype(np.float64)
        r = np.reshape(radiance, (-1, len(wl))).T

        """ normalize illuminant and reflectances """
        illum /= (illum.mean() / 100)
        r /= r.max()

        reflectances = pd.DataFrame(r, columns=list(range(r.shape[1])))
        reflectances.insert(loc=0, column='wl', value=wl)

        illuminant = pd.DataFrame(illum, columns=['default'])
        illuminant.insert(loc=0, column='wl', value=wl)

        return reflectances, illuminant

    class illuminant(object):
        def __init__(self):
            self.cie = self.cie()
            
            self.mls = self.mls()
            self.sfu = self.sfu()

        class cie(object):
            def fl(self):
                df = pd.read_csv(data_folder + illuminant_folder + "cie/fl.csv", index_col=0)
                df[df.columns[1:]]=(df[df.columns[1:]] / (0.01 * df[df.columns[1:]].mean()))
                return df

            def hp(self):
                df = pd.read_csv(data_folder + illuminant_folder + "cie/hp.csv", index_col=0)
                df[df.columns[1:]]=(df[df.columns[1:]] / (0.01 * df[df.columns[1:]].mean()))
                return df

            def std(self):
                return pd.read_csv(data_folder + illuminant_folder + "cie/std.csv", index_col=0)

        
            
                
                
                

            
                
                
                

        class mls(object):
            def mls(self):
                df = pd.read_csv(data_folder + illuminant_folder + "mls/mls.csv", index_col=0)
                df[df.columns[1:]]=(df[df.columns[1:]] / (0.01 * df[df.columns[1:]].mean()))
                return df

        class sfu(object):
            def measured_with_sources(self):
                df = pd.read_csv(data_folder + illuminant_folder + "sfu/measured_with_sources.csv", index_col=0)
                df[df.columns[1:]]=(df[df.columns[1:]] / (0.01 * df[df.columns[1:]].mean()))
                return df

    class reflectances(object):
        def __init__(self):
            self.prerendered = self.prerendered()
            self.monochromatic = self.monochromatic()
            self.babelcolor = self.babelcolor()
            self.chromaxion = self.chromaxion()
            self.sfu = self.sfu()
            self.uef = self.uef()

        class prerendered(object):
            def grid_9(self):
                return [pd.read_csv(data_folder + reflectances_folder + "prerendered/grid_9.csv", index_col=0),
                        'reconstructed, frequency: 9']

            def grid_17(self):
                return [pd.read_csv(data_folder + reflectances_folder + "prerendered/grid_17.csv", index_col=0),
                        'reconstructed, frequency: 17']

            def grid_33(self):
                return [pd.read_csv(data_folder + reflectances_folder + "prerendered/grid_33.csv", index_col=0),
                        'reconstructed, frequency: 33']

            def grid_40(self):
                return [pd.read_csv(data_folder + reflectances_folder + "prerendered/grid_40.csv", index_col=0),
                        'reconstructed, frequency: 40']

            def grid_40_p50_D65(self):
                return [pd.read_csv(data_folder + reflectances_folder + "prerendered/grid_40_p50_D65.csv", index_col=0),
                        'reconstructed, frequency: 40']

        class monochromatic(object):
            def monochromatic(self):
                return [pd.read_csv(data_folder + reflectances_folder + "monochromatic/monochromatic.csv", index_col=0), 'monochromatic']

        class babelcolor(object):
            def sg(self):
                return [pd.read_csv(data_folder + reflectances_folder + "babelcolor/sg.csv", index_col=0), 'SG']

        class chromaxion(object):
            def dc(self):
                return [pd.read_csv(data_folder + reflectances_folder + "chromaxion/dc.csv", index_col=0), 'DC']

        class sfu(object):
            def books(self):
                return [pd.read_csv(data_folder + reflectances_folder + "sfu/books.csv", index_col=0), 'sfu']

            def cardboard(self):
                return [pd.read_csv(data_folder + reflectances_folder + "sfu/cardboard.csv", index_col=0), 'cardboard']

            def cloth(self):
                return [pd.read_csv(data_folder + reflectances_folder + "sfu/cloth.csv", index_col=0), 'cloth']

            def construction(self):
                return [pd.read_csv(data_folder + reflectances_folder + "sfu/construction.csv", index_col=0), 'construction']

            def dupont(self):
                return [pd.read_csv(data_folder + reflectances_folder + "sfu/dupont.csv", index_col=0), 'dupont']

            def krinov(self):
                return [pd.read_csv(data_folder + reflectances_folder + "sfu/krinov.csv", index_col=0), 'krinov']

            def lab_wall(self):
                return [pd.read_csv(data_folder + reflectances_folder + "sfu/lab_wall.csv", index_col=0), 'lab_wall']

            def macbeth(self):
                return [pd.read_csv(data_folder + reflectances_folder + "sfu/macbeth.csv", index_col=0), 'macbeth']

            def munsell(self):
                return [pd.read_csv(data_folder + reflectances_folder + "sfu/munsell.csv", index_col=0), 'munsell']

            def munsell_extended(self):
                return [pd.read_csv(data_folder + reflectances_folder + "sfu/munsell_extended.csv", index_col=0), 'munsell_extended']

            def objects(self):
                return [pd.read_csv(data_folder + reflectances_folder + "sfu/objects.csv", index_col=0), 'objects']

            def paint_chips(self):
                return [pd.read_csv(data_folder + reflectances_folder + "sfu/paint-chips.csv", index_col=0), 'paint-chips']

        class uef(object):
            def __init__(self):
                self.forest390_850_5 = self.forest390_850_5()
                self.paper400_700_10 = self.paper400_700_10()

            class forest390_850_5(object):
                def birch(self):
                    return [pd.read_csv(data_folder + reflectances_folder + "uef/forest390_850_5/birch.csv", index_col=0), 'birch']

                def pine(self):
                    return [pd.read_csv(data_folder + reflectances_folder + "uef/forest390_850_5/pine.csv", index_col=0), 'pine']

                def spruce(self):
                    return [pd.read_csv(data_folder + reflectances_folder + "uef/forest390_850_5/spruce.csv", index_col=0), 'spruce']

            class paper400_700_10(object):
                def cardboardsce(self):
                    return [pd.read_csv(data_folder + reflectances_folder + "uef/paper400_700_10/cardboardsce.csv", index_col=0), 'cardboardsce']

                def cardboardsci(self):
                    return [pd.read_csv(data_folder + reflectances_folder + "uef/paper400_700_10/cardboardsci.csv", index_col=0), 'cardboardsci']

                def mirrorsci(self):
                    return [pd.read_csv(data_folder + reflectances_folder + "uef/paper400_700_10/mirrorsci.csv", index_col=0), 'mirrorsci']

                def newsprintsce(self):
                    return [pd.read_csv(data_folder + reflectances_folder + "uef/paper400_700_10/newsprintsce.csv", index_col=0), 'newsprintsce']

                def newsprintsci(self):
                    return [pd.read_csv(data_folder + reflectances_folder + "uef/paper400_700_10/newsprintsci.csv", index_col=0), 'newsprintsci']

                def papersce(self):
                    return [pd.read_csv(data_folder + reflectances_folder + "uef/paper400_700_10/papersce.csv", index_col=0), 'papersce']

                def papersci(self):
                    return [pd.read_csv(data_folder + reflectances_folder + "uef/paper400_700_10/papersci.csv", index_col=0), 'papersci']

            def munsell380_780_1_glossy(self):
                return [pd.read_csv(data_folder + reflectances_folder + "uef/munsell380_780_1_glossy.csv", index_col=0), 'munsell380_780_1_glossy']

            def munsell380_800_1(self):
                return [pd.read_csv(data_folder + reflectances_folder + "uef/munsell380_800_1.csv", index_col=0), 'munsell380_800_1']

            def natural400_700_5(self):
                return [pd.read_csv(data_folder + reflectances_folder + "uef/natural400_700_5.csv", index_col=0), 'natural400_700_5']

    class spectral_sensitivity(object):
        def __init__(self):
            
            
            

        
            
                

            
                

            
                

        
            
               

        
           
                

        def canon600d(self):
            return [pd.read_csv(data_folder + spectral_sensitivity_folder + "canon600d.csv", index_col=0), 'Canon 600 D']

        def sonydxc930(self):
            return [pd.read_csv(data_folder + spectral_sensitivity_folder + "sonydxc930.csv", index_col=0), 'Sony DXC 930']

        
            

        
            

       
           