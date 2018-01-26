# Yandex-CAPTCHA-Neural-Network-Server
Сервер + обученная нейросеть (opencv вроде) которая разгадывала капчи от яндекс с вероятностью 50% и 3 секунды на одну капчу. До этого была обучена на sklearn и результат был 85% и 15 секунд на капчу что оказалось слишком долго. С 2014 года не трогалась и не развивалась.

**/complete** - директория куда складываются разгаданные капчи
**/data** - директория куда складываются все присланные капчи
**len.py** - мини сервер который угадывает кол-во символов в капче, работает асинхронно. Кол-во потоков равно кол-ву ядер процессора.
**len.xml** - обученная свёрточная нейросеть которая угадывает кол-во символов капче
**sock.py** - сервер эмулирующий работу API сайтов разгадывающих капчи, работает синхронно. Кол-во потоков равно кол-ву ядер процессора умноженное на три (методом тыка найдено)
**sym.py** - мини сервер который угадывает каждый символ в капче, работает асинхронно. Кол-во потоков равно кол-ву ядер процессора.
**sym.xml** - обученная свёрточная нейросеть которая угадывает каждый символ в капче
**words** - все слова которые использовала яндекс капча на момент лета 2014 года 

Разархивируйте нейросеть перед использованием :)