# ViTs_to_EM_simulation
Simulador de campo electromagnetico usando transformers

Para correr localmente
```bash
git clone 
python -m venv vits
source vits/bin/activate
pip install -r requirements.txt
cd ViTs_to_EM_simulation/
```

Para entrenar:
```bash
python main.py --num_sample 27000 --epochs 300 --model_select 2 --batch_size 32 --lr 0.001 --dropout_rate 0.4
```
Para usar debe tener los pesos cargados(pesos en formato pth) y los datos disponibles dentro del .py puede modificar las rutas para eso:
```bash
python generate_samples.py
```


