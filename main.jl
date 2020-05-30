using CSV, JSON, Embeddings, Random
using Flux, Flux.Optimise
include("const.jl")
include("misc.jl")

const embtable = load_embeddings(GloVe, "./glove/glove.840B.300d.txt") # or load_embeddings(FastText_Text) or ...
const get_word_index = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))
labels = collect(keys(LABEL_TO_ID))

m = Chain(
    x -> prepare_batch(x, embtable, get_word_index),
    LSTM(300, 200),
    # Dense(300, 200),
    Dense(200, length(labels)),
    softmax
)

# read datasets
X_train, y_train = load_data("train.json")
X_dev, y_dev = load_data("dev.json")
X_test, y_test = load_data("test.json")

opt = ADAM()

for i in 1:100
    # println(Flux.Tracker.data(loss(preprocess_batch(X_dev), y_dev)))
    @show loss(preprocess_batch(X_dev), y_dev)
    Flux.train!(loss, params(m), [(preprocess_batch(X_train), y_train)], opt)
end

function my_custom_train!(loss, ps, data, opt)
    ps = Params(ps)
    for d in data
        gs = gradient(ps) do
            training_loss = loss(d...)
            # Insert what ever code you want here that needs Training loss, e.g. logging
            return training_loss
        end
        # insert what ever code you want here that needs gradient
        # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge
        update!(opt, ps, gs)
        # Here you might like to check validation set accuracy, and break out to do early stopping
    end
end
