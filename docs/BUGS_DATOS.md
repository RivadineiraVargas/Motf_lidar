# Bugs en la extracción de datos del colega (y su corrección)

Los datos originales (`waymo_10/100/1000` del colega, extraídos con
`save_point_cloud_data.py` — referido como **waymo_antigua**) tenían **dos bugs
independientes** que los volvían en gran parte inutilizables. La versión corregida
es `utilities/save_point_cloud_data_fixed.py` (**waymo_javier**).

---

## Bug 1 — Asociación de objetos rota (`track_index` en vez de `track.id`)

### Qué hacía mal
El script guardaba cada bounding box con un nombre basado en un **contador por
frame**, no en la identidad real del objeto:

```python
# CÓDIGO VIEJO (con bug), ~línea 430:
save_data_to_txt(vertices, os.path.join(subdir, str(track_index) + ".txt"))
#                                                 ^^^^^^^^^^^
#                          posición del objeto en la lista de ESE frame (0,1,2..)
```

`track_index` es "el k-ésimo objeto que aparece en este frame", no la identidad
del auto.

### Por qué corrompe la trayectoria
Cuando un objeto **desaparece** (oclusión, sale del rango del LiDAR), todos los
índices siguientes **se corren una posición**:

```
Frame 10:  auto_A=0   auto_B=1   auto_C=2   auto_D=3
Frame 11:  auto_A=0   [B se ocluye]  auto_C=1   auto_D=2
                                      ^^^^^^^^^
                       C hereda el índice 1, que antes era de B
```

Al reconstruir la "trayectoria del objeto 1" siguiendo los archivos `1.txt` a lo
largo del tiempo, se **cosen pedazos de autos DIFERENTES**. La trayectoria salta
de un auto a otro que puede estar a decenas de metros.

### Síntoma medible
Saltos de **60 a 124 metros entre frames consecutivos** (~0.1s) → implicaría
> 600 m/s (supersónico). Físicamente imposible: una trayectoria Frankenstein.

### Corrección
```python
# CÓDIGO CORREGIDO (save_point_cloud_data_fixed.py):
save_matrix_txt(verts, os.path.join(subdir, f"{track.id}.txt"))
#                                              ^^^^^^^^
#                    identidad ÚNICA y ESTABLE del auto en toda la escena
```
`track.id` (del proto de Waymo) no cambia aunque otros objetos aparezcan o
desaparezcan. Verificado: un track está en el frame 0 y sigue siendo el mismo en
el frame 90.

---

## Bug 2 — Horizonte capado a ~1 segundo

### Qué hacía mal
```python
# CÓDIGO VIEJO (con bug), ~línea 396:
if (frame_i < 11):    # solo los primeros 11 frames
    ...
```

### Por qué importa
Las escenas de Waymo duran **9 s = 91 frames** (10 Hz). El cap tiraba el **88%**
de la trayectoria, dejando ~1 s. Con 1 s de futuro el problema es **trivial**:
el auto va casi en línea recta (sin giros, frenadas ni interacciones). Nuestra
ablación confirmó que a ese horizonte la escena LiDAR no aporta nada.

### Corrección
`save_point_cloud_data_fixed.py` extrae hasta **91 frames (9 s)** con
`max_traj_frames=91`.

---

## Efecto combinado

Para *usar* los datos buggeados había que filtrar (`max_jump=5m`) todo track con
saltos imposibles. El descarte era masivo:

```
waymo_10 (colega):  103 tracks usables  +  256 descartados  →  71% inutilizable
```

Con datos limpios no se descarta nada: de 103 a **1.395 muestras** al mismo
horizonte (13.5×), y además se habilita el horizonte de 9 s.

---

## Tabla resumen

| | waymo_antigua (sucia) | waymo_clean (limpia) |
|---|---|---|
| Identidad del objeto | índice por frame (se corría) | `track.id` persistente |
| Síntoma | saltos 60-124 m (supersónico) | sin saltos |
| Horizonte | 11 frames (~1 s, capado) | 91 frames (~9 s) |
| Tracks utilizables | 29% (71% descartado) | 100% |
| Muestras (mismo horizonte) | 103 | 1.395 |
