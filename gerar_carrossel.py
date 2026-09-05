import base64
import datetime
import json
import os
import re
import subprocess
import time
import unicodedata
import warnings
from io import BytesIO
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageOps
import requests

# Carrega as variáveis de ambiente do arquivo .env seguro
load_dotenv()

# Suprime avisos secundários
warnings.filterwarnings("ignore")

# ================= CONFIGURAÇÕES =================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "ERRO CRÍTICO: A chave GEMINI_API_KEY não foi encontrada no arquivo .env!"
    )

ARQUIVO_JSON = "carrosseis.json"
PASTA_IDENTIDADE = os.path.abspath("identidade")
PASTA_SAIDA = os.path.abspath("saida_carrosseis")
os.makedirs(PASTA_SAIDA, exist_ok=True)


# ================= CARREGAMENTO DA IDENTIDADE =================
def carregar_imagens_identidade():
    imagens = []
    if not os.path.exists(PASTA_IDENTIDADE):
        return imagens
    extensoes = (".png", ".jpg", ".jpeg", ".webp")
    for arq in sorted(os.listdir(PASTA_IDENTIDADE)):
        if arq.lower().endswith(extensoes):
            try:
                img = Image.open(os.path.join(PASTA_IDENTIDADE, arq)).convert(
                    "RGB"
                )
                imagens.append((arq, img))
            except Exception:
                pass
    return imagens


# ================= 1. GERAÇÃO DINÂMICA DE SLIDES =================
def gerar_roteiro_dinamico(client, titulo, lista_identidade):
    print("   [i] Solicitando roteiro de 5 slides ao Gemini...")

    prompt = f"""
Crie o roteiro completo de um carrossel sobre o tema: "{titulo}".

REGRAS CRÍTICAS:
1. TAMANHO: O carrossel terá EXATAMENTE **5 slides**.
2. CONTATOS: NUNCA inclua nenhum tipo de contato ou site.
3. TEXTO: Use pouco texto e bem fáceis para evitar que a IA erre escrita na hora de gerar a imagem.
4. ELEMENTOS: Descreva os elementos SEM EXAGERO PARA NÃO POLUIR que deverão compor a arte de cada slide, como pessoas, objetos, formas e etc, NUNCA REPITA ELEMENTOS DE UM SLIDE PARA O OUTRO.

Retorne EXCLUSIVAMENTE um JSON estruturado desta forma:
{{
  "slides": [
    {{
      "numero": 1,
      "tipo": "capa",
      "headline": "Texto principal curto",
      "subtexto": "Texto secundário curto",
      "elementos": "elementos que devem compor a arte de cada slide."
    }}
  ]
}}
"""

    resp = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.75,
        ),
    )

    match = re.search(r"\{.*\}", resp.text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(resp.text)


# ================= 2. CONVERSORES E EXTRATORES SEGUROS =================
def converter_para_pil(obj_imagem):
    if isinstance(obj_imagem, Image.Image):
        return obj_imagem
    if hasattr(obj_imagem, "image_bytes") and obj_imagem.image_bytes:
        return Image.open(BytesIO(obj_imagem.image_bytes))
    if hasattr(obj_imagem, "_pil_image") and obj_imagem._pil_image:
        return obj_imagem._pil_image
    if isinstance(obj_imagem, bytes):
        return Image.open(BytesIO(obj_imagem))
    raise TypeError(f"Não foi possível converter {type(obj_imagem)} em PIL.Image")


def extrair_imagem_da_resposta(resp):
    parts = getattr(resp, "parts", None)
    if not parts and hasattr(resp, "candidates") and resp.candidates:
        if hasattr(resp.candidates[0], "content") and resp.candidates[0].content:
            parts = getattr(resp.candidates[0].content, "parts", [])

    if parts:
        for part in parts:
            if hasattr(part, "as_image"):
                try:
                    img = part.as_image()
                    if img:
                        return converter_para_pil(img)
                except Exception:
                    pass

            if hasattr(part, "inline_data") and part.inline_data:
                raw_bytes = part.inline_data.data
                if isinstance(raw_bytes, str):
                    raw_bytes = base64.b64decode(raw_bytes)
                return Image.open(BytesIO(raw_bytes))
    return None


# ================= 3. RENDERIZAÇÃO ITERATIVA =================
def gerar_slide_iterativo(client, slide, imagens_anteriores, lista_identidade):
    headline = slide.get("headline", "")
    subtexto = slide.get("subtexto", "")
    elementos = slide.get("elementos", "")

    prompt_iterativo = (
        f"Crie um slide de carrossel vertical na proporção 4:5 para o Instagram, com um design limpo, moderno e profissional.\n\n"
        f"DIRETRIZES DE HIERARQUIA VISUAL (OBRIGATÓRIO):\n"
        f"- O TÍTULO PRINCIPAL DEVE SER GIGANTESCO, EM DESTAQUE ABSOLUTO E OCUPAR DO MEIO PARA CIMA COM LETRAS MAIÚSCULAS MUITO GRANDES E GROSSAS (HEAVY BOLD SANS-SERIF), SENDO O ELEMENTO MAIS IMPORTANTE DA IMAGEM.\n"
        f"- NUNCA DEIXE A IMAGEM POLUÍDA COM MUITOS ELEMENTOS.\n"
        f"- NUNCA REPITA ELEMENTOS DE UM DOS SLIDES PARA O OUTRO.\n"
        f"- NUNCA inclua rótulos como 'capa', 'conteúdo', 'slide 1' ou números estruturais na imagem.\n"
        f"- PROIBIDO absoluto incluir links, sites, e-mails, telefones ou qualquer dado de contato.\n"
        f"- PROIBIDO colocar qualquer tipo de borda ou quadro na imagem.\n"
        f"- Mensagem central inspirada em: '{headline}' e '{subtexto}'. SINTETIZE essa mensagem de forma natural, sem soletração estrita para evitar erros tipográficos.\n"
        f"- Elementos visuais da arte: {elementos}. Dê preferência para elementos reais e não desenhos, como pessoas de verdade.\n"
        f"- Mantenha total harmonia visual, paleta de cores corporativa, identidade da marca e o logotipo 'hiden' integrados de forma fluida com os slides anteriores anexados."
    )

    conteudos = [prompt_iterativo]
    for _, img_pil in lista_identidade:
        conteudos.append(img_pil)
    for img_antiga in imagens_anteriores:
        conteudos.append(img_antiga)

    modelos_imagem = [
        "gemini-3-pro-image-preview",
    ]

    for mod in modelos_imagem:
        try:
            cfg = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="4:5"),
            )
            resp = client.models.generate_content(
                model=mod, contents=conteudos, config=cfg
            )
            img = extrair_imagem_da_resposta(resp)
            if img:
                return img
        except Exception:
            try:
                cfg_simples = types.GenerateContentConfig(
                    response_modalities=["IMAGE"]
                )
                resp = client.models.generate_content(
                    model=mod, contents=conteudos, config=cfg_simples
                )
                img = extrair_imagem_da_resposta(resp)
                if img:
                    return img
            except Exception:
                pass

    raise Exception("Falha ao gerar o slide iterativo.")


# ================= 4. RECORTE EXATO 1080x1350px =================
def salvar_e_recortar_1080x1350(imagem_pil, caminho_saida):
    if imagem_pil.mode in ("RGBA", "P"):
        imagem_pil = imagem_pil.convert("RGB")

    img_ajustada = ImageOps.fit(
        imagem_pil,
        (1080, 1350),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    img_ajustada.save(caminho_saida, format="PNG", quality=95)
    print(
        f"      [OK] Slide salvo e ajustado (1080x1350px): {os.path.basename(caminho_saida)}"
    )


# ================= FUNÇÃO AUXILIAR DE NOTIFICAÇÃO PELO TELEGRAM =================
def enviar_alerta_telegram(titulo):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return

    mensagem = (
        f"*Carrossel Pronto para Aprovação!*\n\n"
        f"📌 *Tema:* {titulo}\n\n"
        f"Acesse para revisão: https://caio-vb.github.io/media/"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(
            url, json={"chat_id": chat_id, "text": mensagem, "parse_mode": "Markdown"}
        )
    except Exception as e:
        print(f"[!] Erro ao enviar notificação no Telegram: {e}")


# ================= FUNÇÃO AUXILIAR DE SYNC COM O GITHUB =================
def enviar_pendencias_git(mensagem_commit="Sincronização de arquivos pendentes"):
    try:
        status_res = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, shell=True
        )
        if status_res.stdout.strip():
            print("[i] Detectados arquivos locais não enviados. Sincronizando com o GitHub...")
            subprocess.run(["git", "add", "."], check=True, shell=True)
            subprocess.run(
                ["git", "commit", "-m", mensagem_commit],
                check=True,
                shell=True,
            )
            subprocess.run(["git", "push"], check=True, shell=True)
            print("[OK] Arquivos pendentes enviados para o GitHub com sucesso!")
    except Exception as e:
        print(f"[!] Aviso no envio automático ao Git: {e}")


# ================= FLUXO PRINCIPAL (MODO DIÁRIO: 1 POR VEZ) =================
def executar():
    # Validação de dias permitidos: Segunda (0), Terça (1) e Sexta (4)
    dia_atual = datetime.datetime.now().weekday()
    dias_permitidos = [0, 1, 5]

    if dia_atual not in dias_permitidos:
        print("[i] Hoje não é dia de geração de carrossel (Apenas Segunda, Terça e Sexta). Encerrando rotina.")
        return

    # 1. Sincroniza com o GitHub antes de começar
    try:
        print("[i] Sincronizando alterações do GitHub (git pull)...")
        subprocess.run(["git", "pull"], check=True, shell=True)
    except Exception as e:
        print(f"[!] Aviso ao executar git pull: {e}")

    if not os.path.exists(ARQUIVO_JSON):
        print(f"Arquivo '{ARQUIVO_JSON}' não encontrado na pasta local.")
        return

    with open(ARQUIVO_JSON, "r", encoding="utf-8") as f:
        carrosseis = json.load(f)

    client = genai.Client(api_key=GEMINI_API_KEY)
    print("[OK] Conectado à API Oficial da Google com o modelo Pro Image.")

    lista_identidade = carregar_imagens_identidade()

    item_alvo = None
    for item in carrosseis:
        if item.get("status") == "pendente":
            item_alvo = item
            break

    if not item_alvo:
        print(
            "\n[i] Todos os carrosséis da fila já foram concluídos! Nenhum item pendente para hoje."
        )
        enviar_pendencias_git("Sincronizacao de pastas locais geradas anteriormente")
        print("\n[SUCESSO] Execução finalizada!")
        return

    # 2. Define o título e gera o slug sanitizado
    titulo = item_alvo["titulo"]

    titulo_limpo = (
        unicodedata.normalize("NFKD", titulo)
        .encode("ASCII", "ignore")
        .decode("utf-8")
    )
    slug = re.sub(r"[^\w\-]", "_", titulo_limpo)
    slug = re.sub(r"_+", "_", slug).strip("_")[:35]

    item_alvo["slug"] = slug
    with open(ARQUIVO_JSON, "w", encoding="utf-8") as f:
        json.dump(carrosseis, f, ensure_ascii=False, indent=2)

    pasta_item = os.path.join(PASTA_SAIDA, slug)

    imagens_prontas = []
    if os.path.exists(pasta_item):
        imagens_prontas = [
            os.path.join(pasta_item, f)
            for f in sorted(os.listdir(pasta_item))
            if f.endswith(".png") and f.startswith("slide_")
        ]

    caminho_copy = os.path.join(pasta_item, "copy.json")
    os.makedirs(pasta_item, exist_ok=True)

    print(f"\n==========================================")
    print(f"Processando Tarefa Diária: {titulo}")
    print(f"==========================================")

    if os.path.exists(caminho_copy):
        with open(caminho_copy, "r", encoding="utf-8") as f:
            dados_copy = json.load(f)
    else:
        dados_copy = gerar_roteiro_dinamico(client, titulo, lista_identidade)
        with open(caminho_copy, "w", encoding="utf-8") as f:
            json.dump(dados_copy, f, ensure_ascii=False, indent=2)

    slides = dados_copy.get("slides", [])
    print(f"[OK] Roteiro estruturado com {len(slides)} slides.")

    print("[2/2] Renderizando slides de forma cumulativa e sequencial...")
    sucessos = 0
    historico_imagens_pil = []

    for img_path in imagens_prontas:
        try:
            historico_imagens_pil.append(Image.open(img_path).convert("RGB"))
        except Exception:
            pass

    for slide in slides:
        num = slide["numero"]
        tipo = slide.get("tipo", "conteudo")
        caminho_img = os.path.join(pasta_item, f"slide_{num}_{tipo}.png")

        if os.path.exists(caminho_img):
            print(f"   -> Slide {num} já existe em disco. Pulando...")
            sucessos += 1
            continue

        print(f"   -> Gerando Slide {num} de {len(slides)} ({tipo})...")
        try:
            img_pil = gerar_slide_iterativo(
                client, slide, historico_imagens_pil, lista_identidade
            )
            salvar_e_recortar_1080x1350(img_pil, caminho_img)
            historico_imagens_pil.append(
                Image.open(caminho_img).convert("RGB")
            )
            sucessos += 1
        except Exception as e:
            print(f"      [ERRO] Falha no Slide {num}: {e}")
            continue

        time.sleep(3)

    if len(slides) > 0 and sucessos == len(slides):
        item_alvo["status"] = "aguardando"
        with open(ARQUIVO_JSON, "w", encoding="utf-8") as f:
            json.dump(carrosseis, f, ensure_ascii=False, indent=2)
        enviar_alerta_telegram(titulo)
        print(f"   [OK] Carrossel diário concluído com sucesso em: {slug}")
    else:
        print(
            f"   [!] Geração incompleta ({sucessos}/{len(slides)}). O status permanecerá pendente para tentar novamente amanhã."
        )

    enviar_pendencias_git(f"Gerado carrossel automato: {titulo}")
    print("\n[SUCESSO] Execução diária finalizada!")


if __name__ == "__main__":
    executar()