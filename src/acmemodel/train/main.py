from .data import load_raw_data
from .model import get_model
from .train import train


def main(optimize: bool = False):
    data = load_raw_data()
    model = get_model()
    train(model, data, optimize=optimize)


if __name__ == '__main__':
    main()
