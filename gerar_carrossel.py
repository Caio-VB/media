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
    print("   [i] Solicitando roteiro rigoroso de 5 slides ao Gemini...")

    prompt = f"""
        Atue como estrategista de conteúdo sênior e crie o roteiro completo de um carrossel de 5 slides sobre o tema: "{titulo}".

        CONTEXTO E POSICIONAMENTO DA MARCA:
        - Especialidade principal: automação de redes sociais, mensagens/atendimento, agendamentos e orçamentos comerciais.
        - Escopo secundário: desenvolvimento de soluções de software e fluxos sob medida para eliminar tarefas manuais.

        DIRETRIZES OBRIGATÓRIAS DE CONTEÚDO:
        1. ESTRUTURA RÍGIDA (EXATAMENTE 5 SLIDES):
        - Slide 1: tipo "capa" (gancho forte, direto e chamativo)
        - Slide 2: tipo "conteudo" (problema operacional ou gargalo)
        - Slide 3: tipo "conteudo" (solução prática / processo automatizado)
        - Slide 4: tipo "conteudo" (resultado obtido: tempo salvo, precisão ou escala)
        - Slide 5: tipo "cta" (chamada final para ação reflexiva ou envio de mensagem)

        2. TEXTO E REDAÇÃO (FOCO EM GERAÇÃO DE IMAGEM):
        - Mantenha textos curtos, impactantes e objetivos.
        - Use vocabulário simples e direto para facilitar a renderização tipográfica sem falhas.
        - PROIBIDO incluir URLs, domínios, e-mails, telefones ou @ de redes sociais no texto.

        3. DIREÇÃO DE ELEMENTOS VISUAIS:
        - Descreva elementos visuais limpos e corporativos (máximo de 2 a 3 elementos por slide, evitando poluição).
        - VARIAÇÃO OBRIGATÓRIA: Não repita os mesmos elementos entre os slides.
        - ESTILO REALISTA: Especifique apenas elementos e contextos reais do ambiente de negócios (ex.: profissionais em escritórios modernos, telas de laptops exibindo gráficos, smartphones com notificações, robótica industrial/tecnológica). Proibido sugerir ilustrações, vetores, 3D cartunesco ou desenhos.

        FORMATO DE SAÍDA:
        Retorne EXCLUSIVAMENTE um objeto JSON válido, sem texto introdutório, sem Markdown extra e sem comentários. Siga estritamente este esquema:
        {{
        "slides": [
            {{
            "numero": 1,
            "tipo": "capa",
            "headline": "Texto principal em poucas palavras",
            "subtexto": "Texto complementar curto",
            "elementos": "Descrição sucinta dos elementos visuais realistas deste slide"
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
        f"Crie um slide de carrossel vertical para Instagram (formato 4:5), ultra-profissional, corporativo e moderno.\n\n"
        f"1. TEXTO E TIPOGRAFIA (RIGOR ABSOLUTO):\n"
        f"- Headline obrigatória: \"{headline}\"\n"
        f"- Subtexto obrigatório: \"{subtexto}\"\n"
        f"- Siga EXATAMENTE os textos fornecidos acima, caractere por caractere. Não resuma, não parafraseie e não adicione palavras.\n"
        f"- O TÍTULO PRINCIPAL (HEADLINE) DEVE SER GIGANTESCO, EM MAIÚSCULAS, FONTE HEAVY BOLD SANS-SERIF, ocupando do meio para cima como o ponto focal mais importante.\n"
        f"- O subtexto deve ser visivelmente menor que a headline.\n"
        f"- Quebre as linhas dos textos de forma harmônica (evite viúvas/órfãs; equilibre o comprimento visual das linhas).\n\n"
        f"2. IDENTIDADE VISUAL E COMPOSIÇÃO:\n"
        f"- Mantenha coerência estética total com as imagens de referência anexadas (paleta de cores, tipografia corporativa e acabamento).\n"
        f"- LOGOTIPO 'hiden': preserve o formato e proporções originais do logotipo anexado. JAMAIS distorça, redimensione incorretamente ou altere seu design.\n"
        f"- Respeite margens seguras: mantenha espaçamento adequado e consistente entre os elementos e as bordas externas do slide.\n"
        f"- Layout clean: composição limpa, sem sobrecarga visual.\n\n"
        f"3. ELEMENTOS VISUAIS E DIREÇÃO DE ARTE:\n"
        f"- Elementos deste slide: {elementos}.\n"
        f"- ESTILO ESTREITO: Utilize exclusivamente elementos fotográficos e realistas (ex.: fotos reais de pessoas em contexto de escritório/tecnologia). Proibido qualquer traço de ilustração vetorial, desenho, 3D cartunesco ou bonecos.\n"
        f"- NÃO REPITA elementos visuais dos slides anteriores anexados; cada slide deve ter sua própria composição única.\n\n"
        f"4. RESTRIÇÕES E PROIBIÇÕES CRÍTICAS (TOLERÂNCIA ZERO):\n"
        f"- PROIBIDO adicionar molduras, bordas externas ou quadros na imagem.\n"
        f"- PROIBIDO incluir números de página, etapas ou rótulos de estrutura (ex.: 'slide 1', 'capa', 'conteúdo', 'cta').\n"
        f"- PROIBIDO incluir links, URLs, domínios de sites, arrobas de redes sociais, e-mails ou números de telefone."
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
    # Validação de dias permitidos: Terça (1), Quinta (3) e Domingo (6)
    dia_atual = datetime.datetime.now().weekday()
    dias_permitidos = [1, 3, 5]

    if dia_atual not in dias_permitidos:
        print("[i] Hoje não é dia de geração de carrossel (Apenas Terça, Quinta e Domingo). Encerrando rotina.")
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

    # 2. Sanitiza o slug
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

    slides = dados_copy.get("slides", [])[:5]
    print(f"[OK] Roteiro validado para {len(slides)} slides.")

    # Mapeamento estrito da nomenclatura oficial
    nomes_arquivos_padrao = {
        1: "slide_1_capa.png",
        2: "slide_2_conteudo.png",
        3: "slide_3_conteudo.png",
        4: "slide_4_conteudo.png",
        5: "slide_5_cta.png",
    }

    imagens_prontas = []
    if os.path.exists(pasta_item):
        imagens_prontas = [
            os.path.join(pasta_item, f)
            for f in sorted(os.listdir(pasta_item))
            if f.endswith(".png") and f.startswith("slide_")
        ]

    print("[2/2] Renderizando slides de forma cumulativa e sequencial...")
    sucessos = 0
    historico_imagens_pil = []

    for img_path in imagens_prontas:
        try:
            historico_imagens_pil.append(Image.open(img_path).convert("RGB"))
        except Exception:
            pass

    for idx, slide in enumerate(slides, start=1):
        nome_esperado = nomes_arquivos_padrao.get(idx, f"slide_{idx}_conteudo.png")
        caminho_img = os.path.join(pasta_item, nome_esperado)

        if os.path.exists(caminho_img):
            print(f"   -> Slide {idx} já existe em disco ({nome_esperado}). Pulando...")
            sucessos += 1
            continue

        print(f"   -> Gerando Slide {idx} de 5 ({nome_esperado})...")
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
            print(f"      [ERRO] Falha no Slide {idx}: {e}")
            continue

        time.sleep(3)

    if sucessos == 5:
        item_alvo["status"] = "aguardando"
        with open(ARQUIVO_JSON, "w", encoding="utf-8") as f:
            json.dump(carrosseis, f, ensure_ascii=False, indent=2)
        enviar_alerta_telegram(titulo)
        print(f"   [OK] Carrossel concluído com exatamente 5 slides em: {slug}")
    else:
        print(
            f"   [!] Geração incompleta ({sucessos}/5). O status permanecerá pendente para tentar novamente."
        )

    enviar_pendencias_git(f"Gerado carrossel automato: {titulo}")
    print("\n[SUCESSO] Execução diária finalizada!")


if __name__ == "__main__":
    executar()