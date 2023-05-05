using SymbolicRegression
import SymbolicRegression.CoreModule: sample_mutation
import SymbolicRegression.MutationFunctionsModule:
    mutate_constant,
    mutate_operator,
    append_random_op,
    prepend_random_op,
    insert_random_op,
    delete_random_op
import Base: setproperty!
import Statistics: std

# Can't pass symbols to Julia, so we have to do string -> symbol conversion:
setproperty!(obj, field::String, val) = setproperty!(obj, Symbol(field), val)

function count_branching(tree::Node{T}) where {T}
    tree.degree == 0 && return 1
    counts = Int[]
    _compute_branching_by_depth!(counts, tree, 1)
    return maximum(counts)
end

function _compute_branching_by_depth!(counts::Vector{Int}, tree, depth::Int)
    length(counts) < depth && push!(counts, 0)
    counts[depth] += 1
    if tree.degree == 0
        return nothing
    elseif tree.degree == 1
        _compute_branching_by_depth!(counts, tree.l, depth + 1)
    else
        _compute_branching_by_depth!(counts, tree.l, depth + 1)
        _compute_branching_by_depth!(counts, tree.r, depth + 1)
    end
    return nothing
end

"""Do a random number of mutations to a tree, mostly insert and add."""
function create_perturbed_tree(
    prev::Node{T}; options, nfeatures, X::AbstractArray{T,2}, max_mutations=5
)::Node{T} where {T}
    weights = MutationWeights(;
        mutate_constant=0.5,
        mutate_operator=1.0,
        optimize=0.0,
        do_nothing=1.0,
        add_node=5.0,
        insert_node=3.0,
        delete_node=1.0,
        simplify=0.0,
        randomize=0.0,
    )
    num_mutations = rand(1:max_mutations)
    tree = [copy_node(prev)]
    for im in 1:num_mutations
        mutation_choice = sample_mutation(weights)
        max_tries = 10
        prev = copy_node(tree[1])
        it = 0
        while it < max_tries
            it += 1
            ctree = copy_node(tree[1])
            ctree = if mutation_choice == :mutate_constant
                temperature = 1.0
                mutate_constant(ctree, temperature, options)
            elseif mutation_choice == :mutate_operator
                mutate_operator(ctree, options)
            elseif mutation_choice == :add_node
                if rand() > 0.5
                    append_random_op(ctree, options, nfeatures)
                else
                    prepend_random_op(ctree, options, nfeatures)
                end
            elseif mutation_choice == :insert_node
                insert_random_op(ctree, options, nfeatures)
            elseif mutation_choice == :delete_node
                if count_nodes(ctree) > 1
                    delete_random_op(ctree, options, nfeatures)
                else
                    ctree
                end
            elseif mutation_choice == :do_nothing
                ctree
            else
                error("Unknown mutation choice: $mutation_choice")
                ctree
            end
            tree[1] = ctree
            test_out, complete = eval_tree_array(ctree, X, options)
            dy_dx =
                (test_out[2:end] .- test_out[1:(end - 1)]) ./
                (X[1, 2:end] .- X[1, 1:(end - 1)])

            normalized_dy_dx = dy_dx ./ (std(test_out) + 1e-8)

            complete &= maximum(abs.(normalized_dy_dx)) < 100
            if complete
                break
            end
        end
        if it == max_tries
            tree[1] = prev
        end
    end
    return tree[1]
end

function just_op_to_string(tree::Node, options::Options)::String
    tree.degree == 0 && error("Not operator.")
    if tree.degree == 1
        op = options.operators.unaops[tree.op]
        op == cos && return "\\cos"
        op == sin && return "\\sin"
        op == relu && return "\\text{relu}"
        op == exp && return "\\exp"
        op == abs && return "\\lvert ~ \\cdot ~ \\rvert"
        op == safe_log && return "\\log"
        return string(op)
    else
        op = options.operators.binops[tree.op]
        op == (+) && return "+"
        op == (-) && return "-"
        op == (*) && return "\\times"
        op == (/) && return "/"
        return string(op)
    end
end