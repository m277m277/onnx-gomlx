package onnx

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/onnx-gomlx/internal/protos"
	"maps"
	"slices"
)

// This file defines the methods that build the computation graph using GoMLX.

// BuildGraph that can be used both for inference and training.
//
// The context in ctx can be set to nil if the model doesn't have any variables.
//
// As in GoMLX graph functions, it panics (throw exceptions) in case of errors.
func (m *Model) BuildGraph(ctx *context.Context, inputs []*Node) (outputs []*Node) {
	// Sanity check of things we don't support yet.
	if len(m.Proto.Functions) > 0 {
		exceptions.Panicf("onnx.BuildGraph does not support ONNX functions")
	}
	if len(m.Proto.Graph.SparseInitializer) > 0 {
		exceptions.Panicf("onnx.BuildGraph does not support ONNX SparseTensors")
	}
	if err := m.ValidateInputs(sliceMap(inputs, func(n *Node) shapes.HasShape { return n })...); err != nil {
		panic(err)
	}

	// ONNX nodes for which we already have a GoMLX node: start with inputs and variables.
	convertedNodes := make(map[string]*Node)
	for inputIdx, nodeName := range m.InputsNames {
		convertedNodes[nodeName] = inputs[inputIdx]
	}

	// Convert all nodes in topological order.
	sortedNodes := m.sortedGraph()
	for _, node := range sortedNodes {
		m.convertNode(ctx, node, convertedNodes)
	}

	// Pick the outputs.
	outputs = make([]*Node, len(m.OutputsNames))
	var found bool
	for outputIdx, nodeName := range m.OutputsNames {
		outputs[outputIdx], found = convertedNodes[nodeName]
		if !found {
			exceptions.Panicf("output node %q not found", nodeName)
		}
	}
	return outputs
}

// sliceMap executes the given function sequentially for every element on in, and returns a mapped slice.
func sliceMap[In, Out any](in []In, fn func(e In) Out) (out []Out) {
	out = make([]Out, len(in))
	for ii, e := range in {
		out[ii] = fn(e)
	}
	return
}

// sortedGraph returns a DAG sorting of the graph, so the returned nodes can be converted in order.
//
// It assumes the inputs and variables are given.
//
// Careful not to mix up node.Name and node.Output (there can be more than one output).
func (m *Model) sortedGraph() []*protos.NodeProto {
	sortedNodes := make([]*protos.NodeProto, 0, len(m.Proto.Graph.Node))

	// Build reverse dependency map.
	outputToDependants := make(map[string]types.Set[*protos.NodeProto])
	for _, node := range m.Proto.Graph.Node {
		for _, input := range node.Input {
			deps, found := outputToDependants[input]
			if !found {
				deps = types.SetWith(node)
				outputToDependants[input] = deps
			} else {
				deps.Insert(node)
			}
		}
	}

	// Check whether node is done.
	doneOutputs := types.MakeSet[string]() // It includes both: Node.Name and Node.Output.
	isReady := func(node *protos.NodeProto) bool {
		for _, input := range node.Input {
			_, found := doneOutputs[input]
			if !found {
				return false
			}
		}
		return true
	}

	// Tabs on finished nodes, and process of marking one node as done.
	nextDoneScan := types.MakeSet[string]()
	markDone := func(outputName string) {
		deps, found := outputToDependants[outputName]
		if !found {
			return
		}
		delete(outputToDependants, outputName)
		for dep := range maps.Keys(deps) {
			if doneOutputs.Has(dep.Name) {
				// This dependant is already marked as done.
				continue
			}
			if !isReady(dep) {
				// This dependant has other dependencies and is not done yet.
				continue
			}
			// One of the dependents is ready, so mark this node as done.
			sortedNodes = append(sortedNodes, dep)
			doneOutputs.Insert(dep.Name)
			for _, output := range dep.Output {
				doneOutputs.Insert(output)
				nextDoneScan.Insert(output)
			}
		}
	}

	// Mark inputs (inputs names and outputs are the same), variables and nodes without any inputs as finished.
	for _, input := range m.InputsNames {
		doneOutputs.Insert(input)
		nextDoneScan.Insert(input)
	}
	for _, node := range m.Proto.Graph.Node {
		if len(node.Input) > 0 {
			continue
		}
		// No inputs: mark as done and append to sortedNodes.
		sortedNodes = append(sortedNodes, node)
		doneOutputs.Insert(node.Name)
		for _, output := range node.Output {
			doneOutputs.Insert(output)
			nextDoneScan.Insert(output)
		}
	}

	// Loop marking nodes as done, and collecting nextDoneScan for the next iteration.
	for len(nextDoneScan) > 0 {
		nextDoneScanSlice := slices.Collect(maps.Keys(nextDoneScan))
		clear(nextDoneScan) // Clear for next batch.
		for _, nodeName := range nextDoneScanSlice {
			markDone(nodeName)
		}
	}
	//fmt.Printf("nodes: %v\n", sliceMap(m.Proto.Graph.Node, func(n *protos.NodeProto) string { return n.Name }))
	//fmt.Printf("sortedNodes: %v\n", sliceMap(sortedNodes, func(n *protos.NodeProto) string { return n.Name }))
	if len(sortedNodes) != len(m.Proto.Graph.Node) {
		exceptions.Panicf("sorting operations graph failed: found %d nodes connected to inputs, but there were %d nodes!?",
			len(sortedNodes), len(m.Proto.Graph.Node))
	}
	return sortedNodes
}

// convertNode converts a single ONNX node to a GoMLX node.
//
// Previously converted nodes are given in convertedNodes.
// The converted output(s) are updated into `convertedNodes`.
//
// It panics (throw exceptions) in case of errors.
func (m *Model) convertNode(ctx *context.Context, node *protos.NodeProto, convertedNodes map[string]*Node) {
	// Convert the node: the usual case is that there is only one output.
	// If res is not nil, it is set to convertedNodes[output[0]].
	// Anything different must be implemented by the specific op switch.
	var res *Node
	inputs := sliceMap(node.Input, func(n string) *Node { return convertedNodes[n] })
	switch node.OpType {
	case "Add":
		res = Add(inputs[0], inputs[1])
	case "MatMul":
		res = MatMul(inputs[0], inputs[1])
	default:
		exceptions.Panicf("unimplemented ONNX node type %q", node.OpType)
	}
	if res != nil {
		convertedNodes[node.Output[0]] = res
	}
}