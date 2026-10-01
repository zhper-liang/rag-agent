import pytest
from jinja2 import StrictUndefined, UndefinedError

from ragagent.prompt.env import RESERVED, attr, build_env, escape_tags


@pytest.mark.parametrize('tag', RESERVED.split('|'))
def test_all_reserved_tags_escaped(tag):
    assert escape_tags(f'<{tag}>x</{tag}>') == f'＜{tag}>x＜/{tag}>'
    assert escape_tags(f'< /{tag.upper()}>') == f'< /{tag.upper()}>'
    assert escape_tags(f'</  {tag.upper()}>') == f'＜/{tag.upper()}>'


def test_plain_code_and_unreserved_tags_preserved():
    code = 'List<String>; a && b; x < 3; <documentary> {{ malicious }}'
    assert escape_tags(code) == code
    env = build_env()
    assert env.from_string('{{ text|esc }}').render(text=code) == code


def test_attr_only_attribute_escaping():
    assert attr('a"<b>&\'') == 'a&quot;&lt;b&gt;&amp;&#x27;'


def test_env_strict_and_autoescape_off():
    env = build_env()
    assert env.autoescape is False and env.undefined is StrictUndefined
    with pytest.raises(UndefinedError):
        env.from_string('{{ missing }}').render()
    assert env.get_template('preprocess_system.j2')
