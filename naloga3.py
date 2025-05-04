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

def meanshift(slika, velikost_okna, dimenzija, max_iter=100, min_cd=5):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    
    # Pridobi dimenzije slike
    h, w, d = slika.shape  # h = višina, w = širina, d = globina (barvni razredi)
    
    # Pretvori sliko v 2D matriko kjer je vsaka vrstica en pixel
    pixels = slika.reshape(-1, d) 
    
    # Če je dimenzija večja od 3, dodaj koordinate (x, y) k barvnim vrednostim
    if dimenzija > 3:
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))  
        coords = np.stack((y_coords, x_coords), axis=-1).reshape(-1, 2)  
        pixels = np.hstack((pixels, coords))  
    
    converged_points = []
    
    # Iterira skozi vse piksle
    for i in range(pixels.shape[0]):
        point = pixels[i]  
        iteration = 0 
        
        while iteration < max_iter:
            # Izračuna razdalje med trenutno točko in vsemi točkami
            distances = np.linalg.norm(pixels - point, axis=1)
            
            # Izračuna uteži z uporabo Gaussovega jedra
            weights = np.exp(-distances**2 / (2 * velikost_okna**2))
            
            # Posodobi točko kot uteženo povprečje vseh točk v okolici
            new_point = np.sum(weights[:, np.newaxis] * pixels, axis=0) / np.sum(weights)
            
            # Preveri konvergenco (če se točka ne premakne več, prekini)
            if np.linalg.norm(new_point - point) < 1e-3:
                break
            
            point = new_point 
            iteration += 1
        
        # Dodaj konvergenčno točko v seznam
        converged_points.append(point)

    # Združi konvergenčne točke v centre glede na min_cd
    centers = []
    for point in converged_points:
        if all(np.linalg.norm(point - center) > min_cd for center in centers):
            centers.append(point)
    
    # Ustvari segmentirano sliko
    labels = np.argmin(np.linalg.norm(pixels[:, np.newaxis] - centers, axis=2), axis=1)
    segmented_pixels = np.array(centers)[labels].astype(np.uint8)
    segmented_image = segmented_pixels.reshape(h, w, d)
    
    return segmented_image


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
    image_path = "slike/peppers.jpg"
    slika = cv.imread(image_path)
    
    if slika is None:
        print(f"Napaka: Slike '{image_path}' ni mogoče naložiti.")
    else:
        slika = cv.cvtColor(slika, cv.COLOR_BGR2RGB)
        
        # Izvedi segmentacijo z uporabo kmeans
        kmeans_segmented = kmeans(slika, k=3, iteracije=10)
        
        # Izvedi segmentacijo z uporabo meanshift
        meanshift_segmented = meanshift(slika, velikost_okna=30, dimenzija=3, max_iter=100, min_cd=5)
        
        # Pretvori slike nazaj v BGR za prikaz 
        original_bgr = cv.cvtColor(slika, cv.COLOR_RGB2BGR)
        kmeans_bgr = cv.cvtColor(kmeans_segmented, cv.COLOR_RGB2BGR)
        meanshift_bgr = cv.cvtColor(meanshift_segmented, cv.COLOR_RGB2BGR)
        
        cv.imshow("Originalna slika", original_bgr)
        cv.imshow("KMeans segmentacija", kmeans_bgr)
        cv.imshow("MeanShift segmentacija", meanshift_bgr)
        
        cv.waitKey(0)
        
        cv.destroyAllWindows()