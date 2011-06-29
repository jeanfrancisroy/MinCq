import numpy as np


class Dataset:
    def __init__(self, X=None, Y=None):
        """ Ensemble de donnees correspondant a un probleme de classification binaire
        
        X : Matrice de vecteur-exemples
        Y : Etiquette des exemples (-1 ou +1)
        """
        self.X = X
        self.Y = Y
    
        
    def loadFromFile(self, filename):
        """ Charge l'ensemble de donnees a partir d'un fichier """
        try:
	        with open(filename) as file:
	            vExample = []
	            vClass   = []
	            
	            for line in file.readlines():
	            	elems = map(float, line.split()) 
	                vExample.append( tuple(elems[1:]) )
	                vClass.append( elems[0] )
	        
                self.X = np.array(vExample)
                self.Y = np.array(vClass)
    	        
                return True
        except:
	        return False
	
                  
    def getNbExamples(self):
        """ Retourne le nombre d'exemples de l'ensemble """
        return np.size(self.X, 0)
        
    
    def getNbFeatures(self):
        """ Retourne le nombre d'attributs d'un exemple"""
        return np.size(self.X, 1)


    def selectExamples(self, indexes):
        """ Conserve les exemples specifies dans la liste indexes """
        self.X = self.X[indexes]
        self.Y = self.Y[indexes]
        
        
    def split(self, trnIndexes):
        """ Divise l'ensemble courant en deux nouveaux ensembles disjoints, dont le premier 
        contient les exemples specifies dans la liste trnIndexes et le deuxieme contient contient 
        les autres exemples """ 
        tstIndexes  = list( set( range(self.getNbExamples() ) ) -set( trnIndexes ) )
        trnData     = Dataset( X=self.X[trnIndexes], Y=self.Y[trnIndexes] )
        tstData     = Dataset( X=self.X[tstIndexes], Y=self.Y[tstIndexes] )
        return (trnData, tstData)
    
    