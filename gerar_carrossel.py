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
        Atue como estrategista sênior de marketing B2B e crie o roteiro completo de um carrossel de exatamente 5 slides sobre o tema: "{titulo}".

        POSICIONAMENTO DA HIDEN SOFTWARE:
        - Especialidades centrais: automação de redes sociais, mensagens/atendimento comercial, agendamentos automáticos e geradores de orçamentos.
        - Escopo secundário: desenvolvimento de scripts e integrações sob medida para eliminar tarefas manuais e devolver tempo para equipes e gestores.

        ESTRUTURA RÍGIDA DE CONVERSÃO (5 SLIDES):
        - Slide 1 (Capa): Gancho forte com dor ou promessa objetiva sobre tempo/custo.
        - Slide 2 (Conteúdo): O gargalo real (onde a empresa gasta dinheiro e tempo desnecessário).
        - Slide 3 (Conteúdo): A virada de chave (como a automação resolve o fluxo).
        - Slide 4 (Conteúdo): O resultado concreto (métricas como horas salvas, velocidade de resposta ou fim do retrabalho).
        - Slide 5 (CTA Comercial): Chamada direta para ação B2B direcionada ao link da bio (ex.: "Clique no link da bio para descobrir onde a sua rotina pode ser automatizada").

        DIRETRIZES DE TEXTO E IMAGEM:
        - TEXTO CURTO E OBJETIVO: Use frases enxutas de alto impacto para garantir leitura rápida no feed e renderização limpa.
        - PROIBIÇÃO DE CONTATOS: Proibido colocar links, sites, e-mails, telefones ou arrobas.
        - ELEMENTOS VISUAIS: Descreva no máximo 2 a 3 elementos visuais por slide. Exija sempre fotos e cenas reais do ambiente corporativo/tecnológico (escritórios modernos, smartphones operando fluxos, dashboards em laptops, pessoas reais focadas).
        - VARIAÇÃO: Não repita os mesmos elementos visuais entre os slides.
        - ESTILO: Proibido sugerir desenhos, vetores, ilustrações 3D ou poses artificiais de banco de imagens.

        Retorne EXCLUSIVAMENTE um JSON estruturado exatamente assim:
        {{
        "slides": [
            {{
            "numero": 1,
            "tipo": "capa",
            "headline": "Headline curta em poucas palavras",
            "subtexto": "Subtexto complementar curto",
            "elementos": "Descrição objetiva de elementos fotográficos realistas"
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
        f"Crie o próximo slide sequencial de um carrossel contínuo vertical (proporção 4:5) para o Instagram, com acabamento premium e corporativo.\n\n"
        f"1. TEXTO E TIPOGRAFIA:\n"
        f'- Headline obrigatória: "{headline}"\n'
        f'- Subtexto obrigatório: "{subtexto}"\n'
        f"- Siga rigorosamente a copy fornecida acima, caractere por caractere. Não altere, não resuma e não crie textos extras.\n"
        f"- HIERARQUIA VISUAL: Headline em letras maiúsculas, Heavy Bold Sans-Serif, dominante na tela.\n"
        f"- O subtexto deve ser visivelmente menor e subordinado à headline.\n"
        f"- Quebre as linhas de texto com equilíbrio estético.\n\n"
        f"2. ALINHAMENTO GEOMÉTRICO E CONTINUIDADE ENTRE SLIDES (REGRA RÍGIDA):\n"
        f"- LINHA DE BASE DO TÍTULO: A primeira linha do título principal DEVE começar EXATAMENTE na mesma coordenada vertical Y (~20% do topo) dos slides anteriores anexados. Nunca desça ou suba a altura do texto de um slide para outro.\n"
        f"- ALINHAMENTO E MARGEM: Mantenha o mesmo recuo lateral (margem esquerda segura de 10%) para os textos em todos os slides.\n"
        f"- POSIÇÃO E ESCALA DO LOGO: O logotipo 'hiden' deve permanecer idêntico em tamanho e posição exata (canto superior esquerdo, respeitando 8% de margem) em relação aos slides anteriores anexados.\n\n"
        f"3. CONTRASTE E CONTINUIDADE CROMÁTICA DO FUNDO:\n"
        f"- O fundo DEVE reproduzir rigorosamente a mesma tonalidade, saturação, temperatura e iluminação dos slides anteriores anexados, garantindo a sensação de um único painel contínuo ao deslizar no feed.\n"
        f"- Fundo escuro e limpo que proporcione legibilidade imediata das fontes claras. É PROIBIDO colocar fundos claros ou trocar a paleta de fundo no meio do carrossel.\n\n"
        f"4. ELEMENTOS VISUAIS E DIREÇÃO DE ARTE:\n"
        f"- Elementos deste slide: {elementos}.\n"
        f"- FOTOGRAFIA REALISTA: Use exclusivamente fotos de alto padrão com pessoas em rotinas reais de trabalho ou dispositivos tecnológicos operacionais. Sem poses forçadas de banco de imagens gratuito.\n"
        f"- PROIBIDO qualquer tipo de vetor, desenho, ilustração, ícones 3D cartunescos ou bordas/molduras.\n"
        f"- Não repita os mesmos elementos visuais presentes nos slides anteriores anexados.\n\n"
        f"5. RESTRIÇÕES RÍGIDAS:\n"
        f"- PROIBIDO incluir molduras, contornos externos ou bordas na arte.\n"
        f"- PROIBIDO adicionar paginação, números de slide ou rótulos ('capa', 'cta', 'slide 2').\n"
        f"- PROIBIDO incluir URLs, arrobas, telefones ou e-mails."
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
def sincronizar_json_com_remoto(item_slug, novo_status):
    """Garante que as aprovações feitas na web não sejam sobrescritas"""
    try:
        subprocess.run(["git", "fetch", "origin", "main"], check=True, shell=True)
        # Tenta ler a versão mais recente do remoto
        res = subprocess.run(
            ["git", "show", "origin/main:carrosseis.json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            shell=True,
        )
        if res.returncode == 0 and res.stdout:
            dados_remotos = json.loads(res.stdout)
            # Atualiza apenas o status do item atual na lista remota
            for r in dados_remotos:
                if r.get("slug") == item_slug or (
                    item_slug
                    and r.get("titulo", "")[:20] in item_slug.replace("_", " ")
                ):
                    r["status"] = novo_status
                    if item_slug:
                        r["slug"] = item_slug
                    break

            with open(ARQUIVO_JSON, "w", encoding="utf-8") as f:
                json.dump(dados_remotos, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[!] Aviso no merge inteligente do JSON: {e}")


def enviar_pendencias_git(mensagem_commit="Sincronização de arquivos pendentes"):
    try:
        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            shell=True,
        )

        if status_res.stdout.strip():
            print(
                "[i] Detectados arquivos locais para envio. Preparando commit..."
            )
            subprocess.run(["git", "add", "."], check=True, shell=True)
            subprocess.run(
                ["git", "commit", "-m", mensagem_commit],
                check=True,
                shell=True,
            )

        # 1. Puxa as alterações remotas usando rebase para evitar commits vazios de merge
        print("[i] Sincronizando alterações remotas (git pull --rebase)...")
        pull_res = subprocess.run(
            ["git", "pull", "--rebase", "origin", "main"],
            capture_output=True,
            text=True,
            shell=True,
        )

        # 2. Se houver conflito no carrosseis.json, resolve priorizando o estado remoto + commit local
        if pull_res.returncode != 0:
            print("[!] Conflito detectado com o GitHub Pages. Resolvendo...")
            subprocess.run(["git", "rebase", "--abort"], check=True, shell=True)

            # Baixa a versão mais recente do remoto mantendo nossa pasta gerada
            subprocess.run(
                ["git", "fetch", "origin", "main"], check=True, shell=True
            )
            subprocess.run(
                ["git", "checkout", "origin/main", "--", ARQUIVO_JSON],
                check=True,
                shell=True,
            )

            # Re-adiciona as imagens e commita
            subprocess.run(["git", "add", "."], check=True, shell=True)
            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"{mensagem_commit} (resolucao de conflito)",
                ],
                check=True,
                shell=True,
            )
            subprocess.run(
                ["git", "pull", "--rebase", "origin", "main"],
                check=True,
                shell=True,
            )

        # 3. Envia com segurança para a branch main
        subprocess.run(
            ["git", "push", "origin", "main"], check=True, shell=True
        )
        print("[OK] Sincronização e push concluídos com sucesso no GitHub!")

    except Exception as e:
        print(f"[!] Erro no envio automático ao Git: {e}")

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
        # Puxa as aprovações que você clicou no painel durante o tempo de geração
        sincronizar_json_com_remoto(slug, "aguardando")
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