# ViTs_to_EM_simulation
Simulador de campo electromagnetico usando transformers

Para ejecutar:
```bash
git clone 
python -m venv vits
source vits/bin/activate
pip install -r requirements.txt
python main.py --num_sample 27000 --epochs 300 --model_select 2 --batch_size 32 --lr 0.001 --dropout_rate 0.3
```
