import cv2 as cv
import numpy as np

def kmeans(slika, k=3, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    
    # Pridobi dimenzije slike
    h, w, d = slika.shape  # h = height, w = width, d = depth (barvni razredi)
    
    # Pretvori sliko v 2D matriko kjer je vsaka vrstica en pixel
    pixels = slika.reshape(-1, d)  # (st_pixlov, depth)
    
    # Inicializira naključno st centrov iz pikslov
    centers = pixels[np.random.choice(pixels.shape[0], k, replace=False)] 

    # Iterira kmeans algoritem
    for _ in range(iteracije):
        # Izračuna Manhattanovo razdaljo med vsakim pixelom in centri
        distances = np.abs(pixels[:, np.newaxis] - centers).sum(axis=2)
        
        # Dodeli vsak pixel najbližjemu centru
        labels = np.argmin(distances, axis=1)
        
        # Posodobi centre kot povprečje vseh pikslov, ki so dodeljeni temu centru
        new_centers = np.array([pixels[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(k)])
        
        # Preveri konvergenco (če se centri ne spreminjajo več, break)
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers  # Posodobi centre za naslednjo iteracijo
    
    # Ustvari segmentirano sliko
    segmented_pixels = centers[labels].astype(np.uint8)  
    segmented_image = segmented_pixels.reshape(h, w, d)  
    
    return segmented_image  

def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    pass

def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    
    # Pridobi dimenzije slike
    h, w, d = slika.shape 
    
    # Pretvori sliko v 2D matriko kjer je vsaka vrstica en pixel
    pixels = slika.reshape(-1, d)  
    
    # Če je dimenzija centra večja od 3, dodaj koordinate (x, y) k barvnim vrednostim
    if dimenzija_centra > 3:
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))  
        coords = np.stack((y_coords, x_coords), axis=-1).reshape(-1, 2) 
        pixels = np.hstack((pixels, coords))

        # Izberi centre naključno, vendar z upoštevanjem praga T
    centers = []  
    while len(centers) < dimenzija_centra:
        candidate = pixels[np.random.choice(pixels.shape[0])]
        
        # Preveri, ali je kandidat dovolj oddaljen od že izbranih centrov
        if all(np.linalg.norm(candidate - center) > T for center in centers):
            centers.append(candidate) 
    
    return np.array(centers) 

if __name__ == "__main__":
    pass