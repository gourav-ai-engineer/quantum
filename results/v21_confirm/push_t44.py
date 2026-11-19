"""Push v21 t44 kernels when a Kaggle GPU slot frees (max 2 concurrent). Polls every 20 min, gives up after 16 h."""
import time
from kaggle.api.kaggle_api_extended import KaggleApi
a = KaggleApi(); a.authenticate()
todo = ["muril_t44", "mbert_t44"]
t0 = time.time()
while todo and time.time() - t0 < 16 * 3600:
    k = todo[0]
    r = a.kernels_push("kaggle/v21_" + k, acc="NvidiaTeslaT4")
    err = getattr(r, "error", None) or ""
    if not err:
        print(time.strftime("%H:%M"), "pushed", k, getattr(r, "url", ""), flush=True)
        todo.pop(0)
        continue
    print(time.strftime("%H:%M"), "waiting:", k, err[:80], flush=True)
    time.sleep(1200)
print("done; not pushed:", todo, flush=True)
