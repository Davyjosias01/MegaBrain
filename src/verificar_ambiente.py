"""Valida o ambiente de desenvolvimento do projeto (Sprint 0).

Confere a versao do Python, a instalacao do PyTorch, a disponibilidade de GPU e
executa operacoes basicas com tensores e autograd. Encerra com codigo de saida
diferente de zero se alguma verificacao falhar.

Uso:
    python src/verificar_ambiente.py
"""

import platform
import sys

VERSAO_PYTHON_MINIMA = (3, 10)

falhas: list[str] = []


def secao(titulo: str) -> None:
    print(f"\n{titulo}")
    print("-" * len(titulo))


def verificar(descricao: str, condicao: bool, detalhe: str = "") -> None:
    marcador = "[OK] " if condicao else "[FALHA] "
    sufixo = f" -- {detalhe}" if detalhe else ""
    print(f"{marcador}{descricao}{sufixo}")
    if not condicao:
        falhas.append(descricao)


secao("Sistema")
print(f"Sistema operacional: {platform.system()} {platform.release()}")
print(f"Arquitetura: {platform.machine()}")
print(f"Executavel Python: {sys.executable}")

secao("Python")
versao = sys.version_info
verificar(
    f"Python {VERSAO_PYTHON_MINIMA[0]}.{VERSAO_PYTHON_MINIMA[1]} ou superior",
    versao >= VERSAO_PYTHON_MINIMA,
    f"encontrado {versao.major}.{versao.minor}.{versao.micro}",
)

secao("PyTorch")
try:
    import torch
except ImportError:
    verificar("PyTorch instalado", False, "execute: pip install -r requirements.txt")
    print("\nAmbiente incompleto: o PyTorch e obrigatorio para as proximas sprints.")
    raise SystemExit(1)

verificar("PyTorch instalado", True, f"versao {torch.__version__}")

cuda_disponivel = torch.cuda.is_available()
if cuda_disponivel:
    print(f"[OK] CUDA disponivel -- {torch.cuda.get_device_name(0)}")
    print(f"     Versao CUDA da build: {torch.version.cuda}")
    memoria_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"     Memoria da GPU: {memoria_gb:.1f} GB")
else:
    print("[AVISO] CUDA indisponivel -- o projeto rodara em CPU (treinamento mais lento)")

dispositivo = torch.device("cuda" if cuda_disponivel else "cpu")

secao("Operacoes com tensores")
torch.manual_seed(123)

tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=dispositivo)
verificar("Criacao de tensores", tensor.shape == (2, 3), f"shape {tuple(tensor.shape)}")

transposta = tensor.T
verificar(
    "Manipulacao de dimensoes (transposicao)",
    transposta.shape == (3, 2),
    f"shape {tuple(transposta.shape)}",
)

produto = tensor @ transposta
esperado = torch.tensor([[14.0, 32.0], [32.0, 77.0]], device=dispositivo)
verificar(
    "Multiplicacao matricial",
    torch.allclose(produto, esperado),
    f"resultado {produto.tolist()}",
)

secao("Autograd")
x = torch.tensor([3.0], requires_grad=True, device=dispositivo)
y = (x**2).sum()  # dy/dx = 2x = 6
y.backward()
verificar(
    "Calculo automatico de gradientes",
    x.grad is not None and torch.allclose(x.grad, torch.tensor([6.0], device=dispositivo)),
    f"dy/dx = {x.grad.item():.1f} (esperado 6.0)",
)

secao("Modulos de rede neural")
modelo = torch.nn.Sequential(
    torch.nn.Linear(3, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1),
).to(dispositivo)

saida = modelo(tensor)
verificar(
    "Construcao e passagem direta de um modelo",
    saida.shape == (2, 1),
    f"saida com shape {tuple(saida.shape)}",
)

parametros = sum(p.numel() for p in modelo.parameters())
# Linear(3, 4) -> 3*4 pesos + 4 vieses = 16; Linear(4, 1) -> 4*1 + 1 = 5
verificar("Contagem de parametros treinaveis", parametros == 21, f"{parametros} parametros")

secao("Resultado")
if falhas:
    print(f"{len(falhas)} verificacao(oes) falharam:")
    for falha in falhas:
        print(f"  - {falha}")
    raise SystemExit(1)

print(f"Todas as verificacoes passaram. Dispositivo em uso: {dispositivo}.")
print("Ambiente pronto para o desenvolvimento do projeto.")
