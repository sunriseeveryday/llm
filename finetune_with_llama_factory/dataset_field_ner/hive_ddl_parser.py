import re
from sqlglot import parse_one, exp


def extract_comments_from_ddl(ddl: str):
    """
    用正则提取字段名和对应的COMMENT文本，返回字典。
    """
    # 捕获形如: 字段名 类型 COMMENT "注释"
    pattern = re.compile(r'(\w+)\s+[^\s,]+(?:\s+[^\s,]+)*\s+COMMENT\s+"([^"]*)"', re.IGNORECASE)
    return dict(pattern.findall(ddl))


def remove_column_comments(ddl: str) -> str:
    """
    去除所有 COMMENT "..." 内容，避免sqlglot解析错误。
    """
    return re.sub(r'COMMENT\s+"[^"]*"', '', ddl, flags=re.IGNORECASE)


def flatten_sqlglot_type(expr, prefix=''):
    """
    递归展开类型结构
    """
    if isinstance(expr, exp.DataType):
        type_this = expr.this

        if type_this == exp.DataType.Type.STRUCT:
            for struct_field in expr.expressions:
                field_name = struct_field.name
                field_type = struct_field.args["kind"]
                yield from flatten_sqlglot_type(field_type, prefix + field_name + ".")
        elif type_this == exp.DataType.Type.ARRAY:
            nested_type = expr.expressions[0]
            yield from flatten_sqlglot_type(nested_type, prefix.rstrip(".") + "[].")
        else:
            yield prefix.rstrip("."), expr
    else:
        yield prefix.rstrip("."), expr


def parse(ddl: str):
    # 先提取注释
    comments = extract_comments_from_ddl(ddl)

    # 去除注释再解析
    ddl_no_comments = remove_column_comments(ddl)
    parsed = parse_one(ddl_no_comments)

    parsed_dict = dict()

    # 输出字段类型 + 注释
    for column in parsed.find_all(exp.ColumnDef):
        col_name = column.name
        col_type = column.args["kind"]
        comment = comments.get(col_name, "")
        for flat_name, flat_type in flatten_sqlglot_type(col_type, prefix=col_name + "."):
            suffix = comment if flat_name == col_name and comment and comment != "-" else ""
            _type = flat_type.sql()
            parsed_dict[flat_name] = [_type, suffix]
    return parsed_dict


if __name__ == "__main__":
    # 读取DDL
    with open("ddl.txt", "r", encoding="utf8") as f:
        ddl = f.read()
    parsed_dict = parse(ddl)
    print()
