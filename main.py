import parselmouth
from parselmouth.praat import call
import numpy as np

def analizar_pitch_formants(ruta_audio, 
                            pitch_floor=75, 
                            pitch_ceiling=600, 
                            max_formant=5500):
    """
    Analiza el archivo de audio para obtener:
    - Pitch promedio (Hz)
    - Formantes promedio (F1, F2, F3, ...)

    Parámetros:
    ----------
    ruta_audio : str
        Ruta del archivo de audio (wav, flac, aiff...)
    pitch_floor : float
        Frecuencia mínima esperada para la detección de pitch (Hz)
    pitch_ceiling : float
        Frecuencia máxima esperada para la detección de pitch (Hz)
    max_formant : float
        Máxima frecuencia en Hz para la búsqueda de formantes (usualmente 5500 para voz masculina/adulta, 
        5000 - 5500 en hombres, 5500 - 6000 en mujeres)
    
    Retorna:
    --------
    (pitch_promedio, f1_promedio, f2_promedio, f3_promedio, ...)
    """

    # Cargar el audio en un objeto Sound de Parselmouth
    sonido = parselmouth.Sound(ruta_audio)

    # 1. Calcular Pitch con Praat (To Pitch)
    # time_step=0.0 => que Praat escoja automáticamente
    # floor y ceiling ajustan el rango de búsqueda
    pitch_obj = call(sonido, "To Pitch", 0.0, pitch_floor, pitch_ceiling)
    n_frames = pitch_obj.get_number_of_frames()

    # Extraer todos los valores de pitch (F0) en cada frame
    pitch_values = []
    for i in range(1, n_frames + 1):
        t = pitch_obj.get_time_from_frame_number(i)
        f0 = pitch_obj.get_value_at_time(t)
        if f0 is not None and not np.isnan(f0):
            pitch_values.append(f0)
    
    if len(pitch_values) > 0:
        pitch_promedio = np.mean(pitch_values)
    else:
        pitch_promedio = 0

    # 2. Calcular Formantes (To Formant (burg))
    # Si la voz es de mujer o muy aguda, podrías subir max_formant a 6000
    formant_obj = call(sonido, "To Formant (burg)", 0.0, 5, max_formant, 0.025, 50)
    
    # Recorremos cada frame para extraer F1, F2, F3...
    f1_list = []
    f2_list = []
    f3_list = []
    # (Puedes agregar F4, F5 si lo requieres)

    # Usamos el mismo número de frames del pitch o
    # podemos usar formant_obj.get_number_of_frames()
    n_formant_frames = formant_obj.get_number_of_frames()

    # Iteramos en cada frame
    for i in range(1, n_formant_frames + 1):
        tiempo = formant_obj.get_time_from_frame_number(i)
        # Extraer F1, F2, F3
        f1 = call(formant_obj, "Get value at time", 1, tiempo, "Hertz", "Linear")
        f2 = call(formant_obj, "Get value at time", 2, tiempo, "Hertz", "Linear")
        f3 = call(formant_obj, "Get value at time", 3, tiempo, "Hertz", "Linear")

        # Podrían salir NaN si no hay voz o no se pudo estimar
        if f1 and not np.isnan(f1): f1_list.append(f1)
        if f2 and not np.isnan(f2): f2_list.append(f2)
        if f3 and not np.isnan(f3): f3_list.append(f3)

    # Promedios de F1, F2, F3
    f1_prom = np.mean(f1_list) if len(f1_list) > 0 else 0
    f2_prom = np.mean(f2_list) if len(f2_list) > 0 else 0
    f3_prom = np.mean(f3_list) if len(f3_list) > 0 else 0

    return pitch_promedio, f1_prom, f2_prom, f3_prom

if __name__ == "__main__":
    ruta = "Boris.wav"  # Cambia a tu archivo
    pitch_floor = 50
    pitch_ceiling = 300
    max_formant = 5500  # Ajusta según la voz

    pitch_prom, f1_prom, f2_prom, f3_prom = analizar_pitch_formants(
        ruta, pitch_floor, pitch_ceiling, max_formant
    )

    print(f"Pitch promedio: {pitch_prom:.2f} Hz")
    print(f"F1 promedio:    {f1_prom:.2f} Hz")
    print(f"F2 promedio:    {f2_prom:.2f} Hz")
    print(f"F3 promedio:    {f3_prom:.2f} Hz")
