import sys
import random
 
##############################################################################
#
# Exemplo:
#   texto = " aa aa, bb cc bb cc aa aa xx.  \n"
#   cadeia = {'aa': ['aa,', 'aa', 'xx.'], 'aa,': ['bb'], 'cc': ['bb', 'aa'], 'bb': ['cc', 'cc']}
#
###############################################################################
 
def cria_cadeia(ngramas=1):
 
    cadeia = {}
 
    for line in sys.stdin:
        line = line.lstrip()
        line = line[::-1].lstrip()[::-1]
 
        splitted_line = line.split(" ")
        splitted_line_list = enumerate(splitted_line)
 
        for index, word in splitted_line_list:
            if word in cadeia.keys():
                if len(splitted_line) > index+1:
                    cadeia[word] = cadeia[word] + [splitted_line[index+1]]
            else:
                if len(splitted_line) > index+1:
                    cadeia[word] = [] + [splitted_line[index+1]]
 
    #print(cadeia)
    return cadeia
 
# def cria_cadeia(ngramas=1):
#     cadeia = {}
#     for line in sys.stdin:
#         line = line.strip().split(' ')
#         for palavra in line[:-1]:
#             new_words = [line[i+1] for i in range(len(line)-1) if line[i] == palavra]
#             if palavra in cadeia: cadeia[palavra] += new_words
#             else: cadeia[palavra] = new_words
#     print(cadeia)
#     return cadeia
 
def gera_sentenca(cadeia, max_termos=1):
    sentenca = []
 
    # Escolhe o primeiro termo aleatoriamente
    termo = random.choice(cadeia.keys())
 
    for i in range(max_termos):
        sentenca.append(termo)
 
        if cadeia.has_key(termo):
            # Escolhe aleatoriamente um dos termos da cadeia para ser a proxima palavra
            termo = random.choice(cadeia[termo])
        else:
            break  # Eh um termo de fim de sentenca e nao precede outros
 
 
    return sentenca
 
 
########################
# Main
########################
 
# Le a primeira linha com os parametros
dados = sys.stdin.readline().strip().split(';')
semente, max_termos, n = int(dados[0]), int(dados[1]), int(dados[2])
#semente = 1
#max_termos = 10
#n = 4
 
random.seed(semente)
 
# Carrega o texto e constroi a cadeia de markov
cadeia = cria_cadeia(1)
# print(cadeia)
 
# Gera as n sentencas
for i in range(n):
    print(' '.join(gera_sentenca(cadeia, max_termos)))