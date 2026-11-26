# Self-Hosted SearXNG Setup (Port 8085)

Self-hosted SearXNG running in Docker on an Azure VM (`127.0.0.1:8085`), accessed locally via SSH tunnel (`8085`) for Python web search and scraping.

---

## 1. VM Setup (One-Time)

```bash
# 1. Create folder & minimal settings.yml (enables JSON + disables rate limiter)
mkdir -p ~/searxng && cd ~/searxng
printf 'use_default_settings: true\nserver:\n  secret_key: "my-super-secret-random-key-987654321"\n  limiter: false\nsearch:\n  formats:\n    - html\n    - json\nengines:\n  - name: google\n    disabled: false\n  - name: bing\n    disabled: false\n  - name: duckduckgo\n    disabled: false\n' > ~/searxng/settings.yml

# 2. Run SearXNG container on VM port 8085
sudo docker run -d \
  --name searxng \
  --restart unless-stopped \
  -p 127.0.0.1:8085:8086 \
  -v /home/ashfaq/searxng:/etc/searxng \
  -e SEARXNG_BASE_URL=http://localhost:8085/ \
  -e SEARXNG_PORT=8086 \
  -e GRANIAN_PORT=8086 \
  -e GRANIAN_HOST=0.0.0.0 \
  docker.io/searxng/searxng:latest

# 3. Verify (should return HTTP/1.1 200 OK)
sleep 4 && curl -i -s "http://localhost:8085/search?q=test&format=json" | head -n 5
```

---

## 2. Local Usage

### Install Dependencies
```bash
pip install requests trafilatura
```

### Start SSH Tunnel (Keep Terminal Open)
```bash
ssh -N -L 8085:localhost:8085 azureuser@<YOUR_VM_IP>
```

### Run Python Script (`search.py`)
```python
import requests
import trafilatura

response = requests.get(
    "http://localhost:8085/search",
    params={
        "q": "site:ais.gov.bd ভুট্টা চাষ বরিশাল",
        "format": "json",
        "engines": "bing",
    },
    timeout=20,
)
response.raise_for_status()

for item in response.json().get("results", [])[:5]:
    url = item["url"]
    full_text = trafilatura.extract(trafilatura.fetch_url(url))
    print(f"Title: {item['title']}\nURL: {url}\nBody:\n{full_text}\n{'-'*60}")
```

---

## 3. Quick VM Commands

* **Restart:** `sudo docker restart searxng`
* **Logs:** `sudo docker logs searxng --tail 20`
* **Remove:** `sudo docker rm -f searxng`
