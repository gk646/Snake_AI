import SnakeGame as game
import multiprocessing as mp
import psutil


def genetic_multi(input_model_name):
    skgame = game.SnakeGame(0)
    skgame1 = game.SnakeGame(0)
    p1 = mp.Process(target=skgame.new_genetic_try, args=(input_model_name, 'new_genetic1', 0))
    p1.start()
    p2 = mp.Process(target=skgame1.new_genetic_try, args=(input_model_name, 'new_genetic2', 0))
    p2.start()
    # print(psutil.virtual_memory())

    # p3 = mp.Process(target=game3.new_genetic_try, args=(input_model_name, 'new_genetic3', high_score))

    # p3.start()
    # time.sleep(1)
    # p4.start()
    # time.sleep(1)
    p1.join()

    p2.join()
    print(psutil.virtual_memory())

    # p3.join()
    # p4.join()

    print("Total Average: " + str(
        (game1.average + game2.average + game3.average + game4.average) / 4))
    f = open('../averages.txt', 'a')
    f.write(
        "\n" + str(
            (game1.average + game2.average + game3.average + game4.average) / 4) + " // " + str(
            game1.average) + "/" +
        str(game2.average) + "/" +
        str(game3.average) + "/" +
        str(game4.average) + "/")

    f.close()


if __name__ == '__main__':
    genetic_multi('new_genetic3')
