using Embeddings
include("const.jl")

function load_data(name)
    content = JSON.parsefile("./data/" * name)
    X, y = [], []
    for js in content
        push!(X, Dict([el => js[el] for el in ["token", "subj_start", "subj_end", "obj_start", "obj_end"]]))
        push!(y, js["relation"])
    end
    X, y
end

function get_embedding(word, embtable, w_ind)
    if word == PAD_TOKEN
        emb = PAD_EMB
    elseif word == UNK_TOKEN
        emb = UNK_EMB
    elseif word == "<SUBJ>"
        emb = SUBJ_EMB
    elseif word == "<OBJ>"
        emb = OBJ_EMB
    else
        ind = get(w_ind, word, 1)
        emb = embtable.embeddings[:,ind]
    end
    return emb
end

function preprocess_batch(batch)
    # TODO preprocess all data as to not waste performance
    max_length = maximum(length.([el["token"] for el in batch]))
    result = []
    for el in batch
        tokens = copy(el["token"])
        tokens[el["subj_start"]+1:el["subj_end"]+1] .= "<SUBJ>"
        tokens[el["obj_start"]+1:el["obj_end"]+1] .= "<OBJ>"
        @assert length(tokens) == length(el["token"])
        push!(result, vcat(tokens, repeat(["<PAD>"], max_length - length(tokens))))
        @assert length(result[end]) == max_length
    end
    result
end

# Negative Log Likelihood Cost
# Instance loss
# loss(xᵢ, yᵢ) = -log(sum(model(xᵢ) .* Flux.onehot(yᵢ, labels)))
# Batch loss
function loss(x, y)
    nll = 0.0

    let preds = m(x)
        onehots = reduce(hcat, [Flux.onehot(y_i, labels) for y_i in y])
        for (pred, ŷ) in zip(eachcol(preds), eachcol(onehots))
               nll += -log(pred' * ŷ)
        end
    end

    Flux.reset!(m)
    nll
end

function prepare_batch(batch, embtable, get_word_index)
    result = []
    for instance in batch
        _temp = [get_embedding(word, embtable, get_word_index) for word in instance]
        push!(result, reduce(hcat, _temp))
    end
    hcat(result...)
end
