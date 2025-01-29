import librosa
import numpy as np
import os

# Filtro de ruido de bajas frecuencias
def high_pass_filter(audio, sample_rate, cutoff=100):
    return librosa.effects.preemphasis(audio, coef=cutoff / sample_rate)

# Cargar el archivo de audio, aplicar filtro y estandarizar
def preprocess_audio(file_path, target_duration=3.0, sr=22050):
    
    audio, sample_rate = librosa.load(file_path, sr=sr)  
    
    audio = high_pass_filter(audio, sample_rate)
    
    target_length = int(target_duration * sample_rate)
    
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')
    
    return audio, sample_rate

# Aplicar STFT y generar espectrograma
def generate_spectrogram(audio, sample_rate, n_fft=2048, hop_length=512):
    try:
        
        stft_result = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    
        spectrogram = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
    
        return spectrogram
    except ValueError as e:
        print(f"Error en la generación de espectrograma: {e}")
    except librosa.util.exceptions.ParameterError as e:
        print(f"Error en parámetros STFT o decibelios para el espectrograma: {e}")
    except Exception as e:
        print(f"Error inesperado en la generación de espectrograma: {e}")


# Define la carpeta de los archivos de audio
folder_path = r'AUDIOS'

# Inicializa listas para almacenar espectrogramas y etiquetas
spectrograms = []
labels = []  
file_names = []
file_names_sorted = sorted(os.listdir(folder_path))

# Procesa cada archivo de audio, verifica que sea un archivo de audio con extensión .wav o .mp3 y genera espectrograma 
for file_name in  file_names_sorted:
    file_path = os.path.join(folder_path, file_name)
    
    if os.path.isfile(file_path) and (file_name.endswith('.wav') or file_name.endswith('.mp3')):
        try:
           
            audio, sample_rate = preprocess_audio(file_path)
            if audio is None:
                print(f"Audio no procesado correctamente: {file_name}")
                continue

            spectrogram = generate_spectrogram(audio, sample_rate)
            if spectrogram is None:
                print(f"Espectrograma no generado correctamente: {file_name}")
                continue
            try:
                # Redimensionar el espectrograma a (96, 96, 1) o cualquier tamaño que necesites
                spectrogram_resized = np.resize(spectrogram, (96, 96))  
                spectrogram_resized = spectrogram_resized.reshape((96, 96, 1))  
            
                spectrograms.append(spectrogram_resized)
                file_names.append(file_name)  
            
                # Asignar etiqueta basándose en el nombre del archivo 
                if "hombre"  in file_name.lower():
                    labels.append(0)  # 0 para hombre
                elif "mujer" in file_name.lower():
                    labels.append(1)  # 1 para mujer
                    
            except ValueError as e:
                print(f"Error al redimensionar el espectrograma para {file_name}: {e}")
            except Exception as e:
                print(f"Error inesperado al redimensionar el espectrograma para {file_name}: {e}")

            # Visualizar el espectrograma (opcional)
            #plt.figure(figsize=(10, 4))
            #librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='log')
            #plt.colorbar(format='%+2.0f dB')
            #plt.title(f'Espectrograma de {file_name}')
            #plt.show()
            #print(f"Procesado y visualizado: {file_name}")
            
        except FileNotFoundError:
            print(f"Archivo no encontrado: {file_name}")
        except Exception as e:
            print(f"Error al procesar {file_name} , debe ser un arcchivo .wav o .mp3 : {e}")


# Getters
def getSpectrogram():
    return np.array(spectrograms)

def getLabels():
    return np.array(labels)

def getFileNames():
    return file_names






