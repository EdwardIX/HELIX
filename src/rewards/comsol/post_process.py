import re
import os
import json
import yaml
import ast

from src.rewards.comsol.gather_materials import add_materials

class ListTransformer(ast.NodeTransformer):
    type_priority = {'JString': 3, 'JDouble': 2, 'JInt': 1}

    def get_element_type(self, node):
        if isinstance(node, ast.List):
            if not node.elts:
                return ('JString', 1)
            child_types = []
            child_depths = []
            for elt in node.elts:
                t, d = self.get_element_type(elt)
                child_types.append(t)
                child_depths.append(d)
            merged_type = max(child_types, key=lambda x: self.type_priority[x])
            merged_depth = max(child_depths) + 1
            return (merged_type, merged_depth)
        elif isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, str):
                return ('JString', 0)
            elif isinstance(val, float):
                return ('JDouble', 0)
            elif isinstance(val, int):
                return ('JInt', 0)
            else:
                return ('JString', 0)
        elif isinstance(node, ast.UnaryOp):  # Type of UnaryOp = Type of Operand
            return self.get_element_type(node.operand)
        else:
            return ('JString', 0)

    def visit_List(self, node):
        element_type, depth = self.get_element_type(node)
        jarray_type = ast.Name(id=element_type, ctx=ast.Load())
        jarray_depth = ast.Constant(value=depth)
        jarray_constructor = ast.Call(
            func=ast.Name(id='JArray', ctx=ast.Load()),
            args=[jarray_type, jarray_depth],
            keywords=[]
        )
        new_list = ast.List(elts=node.elts, ctx=ast.Load())
        new_call = ast.Call(
            func=jarray_constructor,
            args=[new_list],
            keywords=[]
        )
        return new_call


def replace_lists_single_line(code):
    try:
        tree = ast.parse(code)
        transformer = ListTransformer()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)
    except Exception as e:
        return code


def replace_lists(code):
    lines = code.split('\n')
    ret_lines = []
    for line in lines:
        lspace = len(line) - len(line.lstrip())
        ret_lines.append(line[:lspace] + replace_lists_single_line(line.strip()))
    return "\n".join(ret_lines)


class IntToJIntTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.in_unary_op = False

    def visit_UnaryOp(self, node):
        original_in_unary_op = self.in_unary_op
        self.in_unary_op = True
        node.operand = self.visit(node.operand)
        self.in_unary_op = original_in_unary_op

        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int) and not isinstance(
                node.operand.value, bool):
            return ast.Call(
                func=ast.Name(id='JInt', ctx=ast.Load()),
                args=[node],
                keywords=[]
            )
        return node

    def visit_Constant(self, node):
        if not self.in_unary_op and isinstance(node.value, int) and not isinstance(node.value, bool):
            return ast.Call(
                func=ast.Name(id='JInt', ctx=ast.Load()),
                args=[node],
                keywords=[]
            )
        return node

class DictToComsolTransformer():
    def __init__(self, config=None):
        self.config = config
    
    def transform(self, dic): # TODO: support more comsol api
#         return f"""
# model.component('comp1').geom('geom1').create('pol1', 'Polygon')
# model.component('comp1').geom('geom1').feature('pol1').set('source', 'table')
# model.component('comp1').geom('geom1').feature('pol1').set('table', {dic['Polygon']})
# model.component('comp1').geom('geom1').feature('pol1').set('selresult', True)
# model.component('comp1').geom('geom1').feature('pol1').set('selresultshow', 'all')
# model.component('comp1').geom('geom1').create('fil1', 'Fillet')
# model.component('comp1').geom('geom1').feature('fil1').set('radius', {dic['Polygon_Fillet_Radius']})
# model.component('comp1').geom('geom1').feature('fil1').selection('point').set('pol1(1)', {str(dic['Polygon_Fillet_Points'])[1:-1]})
# model.component('comp1').geom('geom1').create('fil2', 'Fillet')
# model.component('comp1').geom('geom1').feature('fil2').set('radius', {dic['Inner_Fillet_Radius']})
# model.component('comp1').geom('geom1').feature('fil2').selection('point').named('r7')
# """
        return f"""
model.component('comp1').geom('geom1').create('pol1', 'Polygon')
model.component('comp1').geom('geom1').feature('pol1').set('source', 'table')
model.component('comp1').geom('geom1').feature('pol1').set('table', {dic['Polygon']})
model.component('comp1').geom('geom1').feature('pol1').set('selresult', True)
model.component('comp1').geom('geom1').feature('pol1').set('selresultshow', 'all')
model.component('comp1').geom('geom1').create('fil2', 'Fillet')
model.component('comp1').geom('geom1').feature('fil2').set('radius', {dic['Inner_Fillet_Radius']})
model.component('comp1').geom('geom1').feature('fil2').selection('point').named('r7')
"""

    def __call__(self, content):
        try:
            return self.transform(json.loads(content))
        except Exception:
            pass

        try:
            return self.transform(yaml.safe_load(content))
        except Exception:
            pass

        return content

def replace_integers_single_line(code):
    try:
        tree = ast.parse(code)
        transformer = IntToJIntTransformer()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)
    except Exception as e:
        return code


def replace_integers(code):
    lines = code.split('\n')
    ret_lines = []
    for line in lines:
        if line.strip().startswith("model."):  # Only Replace Lines in Model Config (Not mph start)
            lspace = len(line) - len(line.lstrip())
            ret_lines.append(line[:lspace] + replace_integers_single_line(line.strip()))
        else:
            ret_lines.append(line)
    return "\n".join(ret_lines)


def replace_constants(code):
    code = re.sub(r"Double\.POSITIVE_INFINITY", r"math.inf", code)
    code = re.sub(r"Double\.NEGATIVE_INFINITY", r"-math.inf", code)
    code = re.sub(r"Double\.NaN", r"math.nan", code)

    return code


def remove_lines(code):
    keywords_to_remove = {
        # r"model\.result\(.+\)\.run\(\);",  # Exclude all plotting command
        r"model\.result",  # Exclude all model.result, include all plotting command
        r"""model\.view\(['"][^'"]+['"]\)""",
        r"""model\.result\(['"][^'"]+['"]\)\.set\(['"]view['"], ['"][^'"]+['"]\)""",  # Remove All views
        r'clearMesh\(\)',
        r'clearMeshes\(\)',
        r'clearSolution\(\)',
        r'clearSolutionData\(\)',  # Remove Clear Solution Options
    }

    ret_lines = []
    for line in code.split('\n'):
        if not any(re.search(keyword, line.strip()) for keyword in keywords_to_remove):
            ret_lines.append(line)

    return '\n'.join(ret_lines)


def process_relative_path(code):
    code = re.sub(r"\.\./Dynamics_and_Vibration/composite_dome_tweeter_eigen.mph", r"composite_dome_tweeter_eigen.mph",
                  code)
    return code


PROBE_TEMPLATE_DEF = """
def add_probe(model):
    list_of_probes = {list_of_probes} """
PROBE_TEMPLATE_CONTENT = """
    for i, (var, dim) in enumerate(list_of_probes):
        if dim == '3d':
            try:
                model.result().numerical().create(f"intvol{i}", "IntVolume")
                model.result().numerical(f"intvol{i}").set("data", "dset1")
                model.result().numerical(f"intvol{i}").selection().all()
                model.result().numerical(f"intvol{i}").setIndex("expr", var, JInt(0))
                value = model.result().numerical(f"intvol{i}").getReal()[-1][-1]
                print(f"Volume Integral Success {var} = {value}")
            except Exception as e:
                print(f"Volume Integral Failed {var}, {e}")
        elif dim == '2d':
            try:
                model.result().numerical().create(f"intsurf{i}", "IntSurface")
                model.result().numerical(f"intsurf{i}").set("intvolume", True)
                model.result().numerical(f"intsurf{i}").set("data", "dset1")
                model.result().numerical(f"intsurf{i}").selection().all()
                model.result().numerical(f"intsurf{i}").setIndex("expr", var, JInt(0))
                value = model.result().numerical(f"intsurf{i}").getReal()[-1][-1]
                print(f"Surface Integral Success {var} = {value}")
            except Exception as e:
                print(f"Surface Integral Failed {var}, {e}")
        elif dim == '1d':
            try:
                model.result().numerical().create(f"intline{i}", "IntLine")
                model.result().numerical(f"intline{i}").set("intsurface", True)
                model.result().numerical(f"intline{i}").set("data", "dset1")
                model.result().numerical(f"intline{i}").selection().all()
                model.result().numerical(f"intline{i}").setIndex("expr", var, JInt(0))
                value = model.result().numerical(f"intline{i}").getReal()[-1][-1]
                print(f"Line Integral Success {var} = {value}")
            except Exception as e:
                print(f"Line Integral Failed {var}, {e}")
        else:
            raise ValueError("Unknown dimension: {dim}")
"""
PROBE_TEMPLATE_RUN = """add_probe(model)"""


def add_probe(code, filename, probe_results):
    assert filename in probe_results, "file not found in probe results, cannot add probe"

    list_of_probes = []
    for k in probe_results[filename].keys():
        mobj = re.match(r"(.*)_([1-3]d)", k)
        assert mobj, "Unknown probe name: {k}"
        list_of_probes.append((mobj.group(1), mobj.group(2)))

    probe_lines = (PROBE_TEMPLATE_DEF.format(list_of_probes=list_of_probes) + PROBE_TEMPLATE_CONTENT).split('\n')
    probe_run_lines = PROBE_TEMPLATE_RUN.split('\n')
    add_indent = (lambda slist, indent=4: [f"{' ' * indent}" + s for s in slist])

    lines = code.split('\n')

    for i, line in enumerate(lines):
        if re.match(r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", line):
            # Find Main Block
            lines = lines[:i] + probe_lines + lines[i:] + add_indent(probe_run_lines)
            break
    else:
        # Main Block Not Found, Add at End
        lines = lines + probe_lines + probe_run_lines

    return "\n".join(lines)


class PostProcessor:
    def __init__(self, config={}):
        self.config = config
        if self.config.get('add_materials', True):
            if self.config.get('matlib', None) is None:
                with open(self.config['matlib_path'], 'r') as f:
                    self.matlib = json.load(f)
            else:
                self.matlib = self.config['matlib']

        if self.config.get('add_probe', True):
            if self.config.get('probe_results', None) is None:
                with open(self.config['probe_result_path'], 'r') as f:
                    self.probe_results = json.load(f)
            else:
                self.probe_results = self.config['probe_results']

    def __call__(self, code, filename):

        if self.config.get('remove_lines', True):
            code = remove_lines(code)

        if self.config.get('replace_constants', True):
            code = replace_constants(code)

        if self.config.get('process_relative_path', True):
            code = process_relative_path(code)

        if self.config.get('add_materials', True):
            code = add_materials(code, filename, self.matlib)

        if self.config.get('replace_lists', True):
            code = replace_lists(code)

        if self.config.get('replace_integers', True):
            code = replace_integers(code)

        if self.config.get('add_probe', True):
            code = add_probe(code, filename, self.probe_results)

        return code


config = {
    'input_folder': '/data/SciLLM/comsol/pymodels/v6.3out',
    'output_folder': '/data/SciLLM/comsol/pymodels/v6.3out_post_probe',
    'post_process_config': {
        'replace_lists': True,
        'replace_integers': True,
        'replace_constants': True,
        'process_relative_path': True,
        'add_materials': True,
        'remove_lines': True,
        'matlib': None,
        'matlib_path': '/data/SciLLM/comsol/pymodels/mlib.json',
        'add_probe': True,
        'probe_result': None,
        'probe_result_path': '/data/SciLLM/comsol/pymodels/probes_results.json',
    }
}

if __name__ == '__main__':
    for root, _, files in os.walk(config["input_folder"]):
        java_files = [f for f in files if f.endswith(".py")]
        if not java_files:
            continue

        print(f"Processing directory: {root}")
        relative_path = os.path.relpath(root, config["input_folder"])
        output_path = os.path.join(config["output_folder"], relative_path)
        post_process = PostProcessor(config['post_process_config'])
        os.makedirs(output_path, exist_ok=True)
        for file in files:
            print(f"  Processing file: {file}")
            with open(os.path.join(root, file), 'r') as f:
                code = f.read()
                # code = post_process(code, os.path.join(relative_path, file.replace(".py", ".java")))
                code = post_process(code, file.replace(".py", ""))
            with open(os.path.join(output_path, file), 'w') as f:
                f.write(code)