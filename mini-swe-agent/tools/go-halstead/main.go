// go-halstead: compute per-function Halstead metrics and CC for Go files.
// Uses only stdlib (go/parser, go/ast, go/token, go/scanner) — no import resolution needed.
package main

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"math"
	"os"
)

func isOperator(tok token.Token) bool {
	return tok.IsOperator() || tok.IsKeyword()
}

func isOperand(tok token.Token) bool {
	return tok == token.IDENT || tok == token.INT || tok == token.FLOAT ||
		tok == token.IMAG || tok == token.CHAR || tok == token.STRING
}

type halstead struct {
	N1       int
	N2       int
	n1Unique map[string]struct{}
	n2Unique map[string]struct{}
}

func newHalstead() *halstead {
	return &halstead{n1Unique: map[string]struct{}{}, n2Unique: map[string]struct{}{}}
}

func (h *halstead) add(tok token.Token, lit string) {
	if isOperator(tok) {
		h.n1Unique[tok.String()] = struct{}{}
		h.N1++
	} else if isOperand(tok) {
		key := lit
		if key == "" {
			key = tok.String()
		}
		h.n2Unique[key] = struct{}{}
		h.N2++
	}
}

// tokenizeSnippet scans a raw Go source snippet (function body bytes).
func tokenizeSnippet(snippet []byte) *halstead {
	h := newHalstead()
	fset := token.NewFileSet()
	file := fset.AddFile("", fset.Base(), len(snippet))
	var s scanner.Scanner
	s.Init(file, snippet, nil, 0)
	for {
		_, tok, lit := s.Scan()
		if tok == token.EOF {
			break
		}
		if tok == token.ILLEGAL {
			continue
		}
		h.add(tok, lit)
	}
	return h
}

func countCC(body *ast.BlockStmt) int {
	cc := 1
	ast.Inspect(body, func(n ast.Node) bool {
		switch v := n.(type) {
		case *ast.IfStmt, *ast.ForStmt, *ast.RangeStmt, *ast.SelectStmt:
			cc++
		case *ast.CaseClause:
			if v.List != nil {
				cc++
			}
		case *ast.CommClause:
			if v.Comm != nil {
				cc++
			}
		case *ast.BinaryExpr:
			if v.Op == token.LAND || v.Op == token.LOR {
				cc++
			}
		}
		return true
	})
	return cc
}

type FuncResult struct {
	Name       string  `json:"name"`
	N1         int     `json:"N1"`
	N2         int     `json:"N2"`
	N1Unique   int     `json:"n1"`
	N2Unique   int     `json:"n2"`
	Vocabulary int     `json:"vocabulary"`
	Length     int     `json:"length"`
	Volume     float64 `json:"volume"`
	Difficulty float64 `json:"difficulty"`
	Effort     float64 `json:"effort"`
	MI         float64 `json:"mi"`
	CC         int     `json:"cc"`
}

type Result struct {
	Functions []FuncResult `json:"functions"`
}

func rnd(v float64) float64 { return math.Round(v*1000) / 1000 }

func main() {
	if len(os.Args) < 2 {
		os.Stderr.WriteString("usage: go-halstead <file.go>\n")
		os.Exit(1)
	}
	src, err := os.ReadFile(os.Args[1])
	if err != nil {
		os.Stderr.WriteString("read error: " + err.Error() + "\n")
		os.Exit(1)
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, os.Args[1], src, parser.AllErrors)
	if f == nil {
		os.Stderr.WriteString("parse error: " + err.Error() + "\n")
		os.Exit(1)
	}

	result := Result{Functions: []FuncResult{}}

	for _, decl := range f.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}

		name := fd.Name.Name
		if fd.Recv != nil && len(fd.Recv.List) > 0 {
			switch t := fd.Recv.List[0].Type.(type) {
			case *ast.StarExpr:
				if id, ok2 := t.X.(*ast.Ident); ok2 {
					name = id.Name + "." + name
				}
			case *ast.Ident:
				name = t.Name + "." + name
			}
		}

		tf := fset.File(fd.Body.Lbrace)
		startOff := tf.Offset(fd.Body.Lbrace)
		endOff := tf.Offset(fd.Body.Rbrace)
		if endOff > len(src) {
			endOff = len(src)
		}
		snippet := src[startOff:endOff]

		h := tokenizeSnippet(snippet)
		n1 := len(h.n1Unique)
		n2 := len(h.n2Unique)
		vocab := n1 + n2
		length := h.N1 + h.N2

		var vol, diff, effort, mi float64
		mi = 100
		if vocab > 0 && length > 0 {
			vol = float64(length) * math.Log2(float64(vocab))
			if n2 > 0 {
				diff = float64(n1) / 2.0 * float64(h.N2) / float64(n2)
			}
			effort = diff * vol
			if vol > 0 {
				mi = math.Max(0, (171-5.2*math.Log(vol))*100/171)
			}
		}

		cc := countCC(fd.Body)

		result.Functions = append(result.Functions, FuncResult{
			Name:       name,
			N1:         h.N1,
			N2:         h.N2,
			N1Unique:   n1,
			N2Unique:   n2,
			Vocabulary: vocab,
			Length:     length,
			Volume:     rnd(vol),
			Difficulty: rnd(diff),
			Effort:     rnd(effort),
			MI:         rnd(mi),
			CC:         cc,
		})
	}

	json.NewEncoder(os.Stdout).Encode(result)
}
