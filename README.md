# ndviforecasting

Список фвйлов:

Main_NDVI.m: основной скрипт, вызывает методы предобработки данных (поэтапный регрессионный анализ и метод главных компонент), обучения модели Layer Recurrent Neural Network (обобщенную версию рекуррентной сети Элмана) и проверки обученной модели;

trainLRNN.m: обучение модели нейронной сети Layer Recurrent Neural Network (LRNN);

testLRNN.m: проверка обученной модели LRNN.


Для вызова функции compute_mapping (метод главных компонент) надо инсталировать инструментарий Matlab Toolbox for Dimensionality Reduction: https://lvdmaaten.github.io/drtoolbox/.

