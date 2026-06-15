# -*- coding: utf-8 -*-
"""v5 情感分析 Gradio 演示界面（汇报视频录屏用）"""
import gradio as gr

from test import predict, label_names

# 情感标签配色（界面展示用）
LABEL_STYLE = {
    '积极': ('#16a34a', '😊'),
    '中性': ('#ca8a04', '😐'),
    '消极': ('#dc2626', '😞'),
}

EXAMPLES = [
    ['今天心情特别好，太开心了！'],
    ['服务态度极差，再也不会买了'],
    ['今天天气一般，正常上班'],
    ['真的气死我了，什么垃圾东西'],
    ['就这样吧，没什么感觉'],
    ['一般般，没有惊喜也没有失望'],
    ['这家店味道不错，推荐！'],
    ['爱了爱了，无限回购！'],
]


def analyze(text):
    """分析单句情感，返回 Gradio 输出组件所需格式"""
    text = (text or '').strip()
    if not text:
        return (
            '<p style="color:#6b7280;font-size:18px;">请输入待分析的微博文本</p>',
            '',
        )

    label, probs = predict(text)
    color, emoji = LABEL_STYLE.get(label, ('#2563eb', '•'))

    label_html = (
        f'<div style="text-align:center;padding:24px;">'
        f'<div style="font-size:20px;color:#6b7280;margin-bottom:8px;">预测情感</div>'
        f'<div style="font-size:42px;font-weight:bold;color:{color};">'
        f'{emoji} {label}'
        f'</div></div>'
    )

    prob_lines = []
    for i, name in enumerate(label_names):
        pct = probs[i] * 100
        bar_color = LABEL_STYLE[name][0]
        highlight = 'font-weight:bold;' if name == label else ''
        prob_lines.append(
            f'<div style="margin:10px 0;">'
            f'<div style="display:flex;justify-content:space-between;{highlight}">'
            f'<span>{name}</span><span>{pct:.1f}%</span></div>'
            f'<div style="background:#e5e7eb;border-radius:6px;height:14px;overflow:hidden;">'
            f'<div style="width:{pct:.1f}%;background:{bar_color};height:100%;"></div>'
            f'</div></div>'
        )
    prob_html = (
        '<div style="padding:8px 4px;">'
        + ''.join(prob_lines)
        + '</div>'
    )

    return label_html, prob_html


def build_demo():
    """构建 Gradio 界面"""
    with gr.Blocks(title='微博情感分析 v5') as demo:
        gr.Markdown(
            '# 微博三分类情感分析\n'
            '输入文本后点击「分析」，或选择下方示例快速体验。'
        )
        with gr.Row():
            with gr.Column(scale=1):
                text_in = gr.Textbox(
                    label='输入文本',
                    placeholder='请输入微博或评论内容…',
                    lines=4,
                )
                analyze_btn = gr.Button('分析', variant='primary')
                gr.Examples(examples=EXAMPLES, inputs=text_in, label='演示示例')
            with gr.Column(scale=1):
                label_out = gr.HTML(label='预测结果')
                prob_out = gr.HTML(label='三类概率')

        analyze_btn.click(analyze, inputs=text_in, outputs=[label_out, prob_out])
        text_in.submit(analyze, inputs=text_in, outputs=[label_out, prob_out])
    return demo


if __name__ == '__main__':
    app = build_demo()
    app.launch(
        server_name='127.0.0.1',
        server_port=7860,
        inbrowser=True,
        show_error=True,
    )
