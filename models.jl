module Models
    export Attention

    using Flux

    struct Attention
        dense1
        lstm
        attnᵢ
        attn_query
        v
        dense2
    end

    @Flux.functor Attention

    function (attn::Attention)(xᵢ)
        # Weight matrix for attention instead of dense layer? -> no bias
        in  = attn.dense1(xᵢ)
        h   = attn.lstm(in)
        q   = attn.attn_query(h[:, end])
        αₙ  = softmax(attn.v * tanh.(attn.attnᵢ(h) .+ q), dims=2)
        hₛ   = reduce(+, eachcol(αₙ .* h))
        out = attn.dense2(hₛ)
        softmax(out)
    end
end
