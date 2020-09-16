## Convolução de Imagem

### Atividade Proposta:
- [x] Implementar a convolução de uma imagem I com um núcleo K
- [x] Núcleo Gaussiano (sigma = 1, 2, 3)
- [x] Detector de bordas de Sobel 

### Modo de Usar
###### Parâmetros:
* ```--imageIn -i``` : caminho da imagem de entrada
* ```--filter -f```: filtro a ser utilizado, tendo como opções ``gaussian`` ou ``sobel``
* ```--sigma -s```: ao escolher o filtro ``gaussian``, é necessário inserir o parâmetro sigma para calcular o kernel, tendo como opções ``1, 2 ou 3``

###### Exemplo:
```
  python imageConvolution.py -i 'images/building.jpg' -f sobel
```
![Alt Text](https://github.com/CamAndrade/imageConvolution/blob/master/results/sobel.png)

```
  python imageConvolution.py -i 'images/noisy.jpg' -f gaussian -s 3
```
![Alt Text](https://github.com/CamAndrade/imageConvolution/blob/master/results/gaussian.png)
