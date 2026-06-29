from tsdalia.functions import load_dataset
from tsdalia.anomaly_detector import (
    TransformerAutoEncoderDetector,
    BasicAutoEncoderDetector,
    DiffusionDetector,
    BenchmarkDetectors,
)


if __name__ == "__main__":

    trainX, testX, testY = load_dataset()

    # tae = TransformerAutoEncoderDetector(
    #     window_size=100, ts_dims=3, batch_size=128, lr=0.00001
    # )
    # train_x, test_x, test_y = tae.create_windows(
    #     train_x=trainX, test_x=testX, test_y=testY
    # )
    # tae.fit(train_x=train_x, num_epochs=20)
    # tae.predict(test_x=test_x)
    # tae.evaluate(test_y=test_y)

    # dm = DiffusionDetector(
    #     window_size=120,
    #     lr=0.001,
    #     batch_size=32,
    #     noise_steps=50,
    #     denoise_steps=100,
    #     diff_lambda=0.05,
    # )
    # train_x, test_x, test_y = dm.create_windows(
    #     train_x=trainX, test_x=testX, test_y=testY
    # )
    # dm.fit(train_x=train_x, num_epochs=20)
    # dm.predict(test_x=test_x)
    # dm.evaluate(test_y=test_y)

    # bae = BasicAutoEncoderDetector()
    # train_x, test_x, test_y = bae.create_windows(
    #     train_x=trainX, test_x=testX, test_y=testY
    # )
    # bae.fit(train_x=train_x, num_epochs=20)
    # bae.predict(test_x=test_x)
    # bae.evaluate(test_y=test_y)

    bench = BenchmarkDetectors(outliers_fraction=0.05)
    train_x, test_x, test_y = bench.create_windows(
        train_x=trainX, test_x=testX, test_y=testY, dataloader=False
    )
    bench.fit(train_x=train_x)
    bench.predict(test_x=test_x)
    bench.evaluate(test_y=test_y)
